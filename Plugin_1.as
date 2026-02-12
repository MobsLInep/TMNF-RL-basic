bool g_pluginEnabled = true;
int g_streamInterval = 10;
int g_lastStreamTime = 0;
int g_lastRaceTime = -1;
int g_lastValidRaceTime = 0;
bool g_finishHandled = false;
Net::Socket@ g_socket = null;
Net::Socket@ g_clientSocket = null;

enum ControlMessageType {
    CSetInputState = 1,
    CResetRace = 2
}

void Main()
{
    @g_socket = Net::Socket();
    bool listenSuccess = g_socket.Listen("127.0.0.1", 5555);
    if (listenSuccess) {
        log("✓ Socket listening on 127.0.0.1:5555");
    } else {
        log("✗ Failed to listen on 127.0.0.1:5555");
    }
    
    RegisterVariable("plugin_enabled", true);
    RegisterVariable("stream_interval_ms", 10);
}

void OnDisabled()
{
    if (g_socket !is null) {
        @g_socket = null;
    }
    if (g_clientSocket !is null) {
        @g_clientSocket = null;
    }
}

PluginInfo@ GetPluginInfo()
{
    auto info = PluginInfo();
    info.Name = "Car Data Socket + Input Control + Auto-Restart";
    info.Author = "Data Streamer";
    info.Version = "v5.9.1";
    info.Description = "Streams telemetry + receives input commands + reset support";
    return info;
}

void Render()
{
    if (g_socket is null) return;
    
    Net::Socket@ newClient = g_socket.Accept(0);
    if (newClient !is null) {
        @g_clientSocket = newClient;
        newClient.NoDelay = true;
        log("✓ Client connected (IP: " + g_clientSocket.RemoteIP + ")");
    }
}

void OnGameStateChanged(TM::GameState state)
{
}

void OnCommandListChanged(CommandList@ prev, CommandList@ current, CommandListChangeReason reason)
{
}

void PollControlMessages(SimulationManager@ simManager)
{
    if (g_clientSocket is null) return;
    if (g_clientSocket.Available == 0) return;
    
    try {
        int msgType = g_clientSocket.ReadInt32();
        
        if (msgType == ControlMessageType::CSetInputState) {
            bool left       = g_clientSocket.ReadUint8() > 0;
            bool right      = g_clientSocket.ReadUint8() > 0;
            bool accelerate = g_clientSocket.ReadUint8() > 0;
            bool brake      = g_clientSocket.ReadUint8() > 0;
            
            if (simManager.InRace) {
                simManager.SetInputState(InputType::Left,  left ? 1 : 0);
                simManager.SetInputState(InputType::Right, right ? 1 : 0);
                simManager.SetInputState(InputType::Up,    accelerate ? 1 : 0);
                simManager.SetInputState(InputType::Down,  brake ? 1 : 0);
            }
        }
        else if (msgType == ControlMessageType::CResetRace) {
            // Handle reset command from Python
            if (simManager.InRace) {
                log("✓ Reset command received");
                simManager.PreventSimulationFinish();
                ExecuteCommand("save_replay");
                ExecuteCommand("press delete");
            }
        }
    } catch {
        log("✗ Client disconnected");
        @g_clientSocket = null;
    }
}

void SendDataToSocket(string jsonData)
{
    if (g_clientSocket is null) {
        return;
    }
    
    string dataWithNewline = jsonData + "\n";
    bool writeSuccess = g_clientSocket.Write(dataWithNewline);
    
    if (!writeSuccess) {
        log("✗ Client disconnected");
        @g_clientSocket = null;
    }
}

void StreamCarDataToSocket(SimulationManager@ simManager)
{
    g_pluginEnabled = GetVariableBool("plugin_enabled");
    g_streamInterval = int(GetVariableDouble("stream_interval_ms"));
    
    if (!g_pluginEnabled) return;
    
    int currentTime = simManager.TickTime;
    
    if (currentTime < g_lastStreamTime + g_streamInterval) {
        return;
    }
    
    g_lastStreamTime = currentTime;
    
    vec3 position = simManager.Dyna.CurrentState.Location.Position;
    vec3 velocity = simManager.Dyna.CurrentState.LinearSpeed;
    float speedMs = velocity.Length();
    float speedKmh = speedMs * 3.6f;
    
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    simManager.Dyna.CurrentState.Location.Rotation.GetYawPitchRoll(yaw, pitch, roll);
    
    int stuntsScore = simManager.PlayerInfo.StuntsScore;
    
    vec3 angularVel = simManager.Dyna.CurrentState.AngularSpeed;
    float turningRate = angularVel.y;
    
    int curCheckpoint = simManager.PlayerInfo.CurCheckpointCount;
    int totalCheckpoints = simManager.PlayerInfo.Checkpoints.Length;
    
    string json = "{";
    json += "\"time_ms\":" + currentTime + ",";
    json += "\"position\":{\"x\":" + FormatFloat(position.x) + ",\"y\":" + FormatFloat(position.y) + ",\"z\":" + FormatFloat(position.z) + "},";
    json += "\"velocity\":{\"x\":" + FormatFloat(velocity.x) + ",\"y\":" + FormatFloat(velocity.y) + ",\"z\":" + FormatFloat(velocity.z) + "},";
    json += "\"speed_kmh\":" + FormatFloat(speedKmh) + ",";
    json += "\"rotation\":{\"yaw\":" + FormatFloat(yaw) + ",\"pitch\":" + FormatFloat(pitch) + ",\"roll\":" + FormatFloat(roll) + "},";
    json += "\"stunts_score\":" + stuntsScore + ",";
    json += "\"turning_rate\":" + FormatFloat(turningRate) + ",";
    json += "\"checkpoint\":{\"current\":" + curCheckpoint + ",\"total\":" + totalCheckpoints + "}";
    json += "}";
    
    SendDataToSocket(json);
}

string FormatFloat(float value)
{
    int intPart = int(value);
    int fracPart = int((value - intPart) * 1000);
    if (fracPart < 0) fracPart = -fracPart;
    
    string result = "" + intPart;
    if (fracPart > 0) {
        result += "." + fracPart;
    }
    return result;
}

void OnRunStep(SimulationManager@ simManager)
{
    g_pluginEnabled = GetVariableBool("plugin_enabled");
    if (!g_pluginEnabled) return;
    
    int raceTime = simManager.RaceTime;
    
    // Track the last valid (non-negative) race time while racing
    if (raceTime >= 0) {
        g_lastValidRaceTime = raceTime;
    }
    
    // Detect race started: raceTime goes from < 0 to >= 0
    if (g_lastRaceTime < 0 && raceTime >= 0) {
        log("✓ Race started");
        g_lastStreamTime = 0;
        g_lastValidRaceTime = 0;
        g_finishHandled = false;
    }
    
    g_lastRaceTime = raceTime;
    
    if (raceTime >= 0) {
        PollControlMessages(simManager);
        StreamCarDataToSocket(simManager);
    }
}

void OnSimulationBegin(SimulationManager@ simManager)
{
    g_lastStreamTime = 0;
    g_lastRaceTime = 0;
    g_lastValidRaceTime = 0;
    g_finishHandled = false;
}

void OnSimulationStep(SimulationManager@ simManager, bool userCancelled)
{
    g_pluginEnabled = GetVariableBool("plugin_enabled");
    if (!g_pluginEnabled || userCancelled) return;
    
    PollControlMessages(simManager);
    StreamCarDataToSocket(simManager);
}

void OnSimulationEnd(SimulationManager@ simManager, SimulationResult result)
{
    g_lastStreamTime = 0;
    g_lastRaceTime = -1;
    g_lastValidRaceTime = 0;
    g_finishHandled = false;
    g_pluginEnabled = GetVariableBool("plugin_enabled");
}

void OnCheckpointCountChanged(SimulationManager@ simManager, int current, int target)
{
    // Send normal checkpoint event
    string eventJson = "{\"event\":\"checkpoint_passed\",\"current\":" + current + ",\"total\":" + target + "}";
    SendDataToSocket(eventJson);
    
    // Detect finish ONCE when we hit the final checkpoint
    if (!g_finishHandled && current == target) {
        // Prevent game from finishing the simulation (stays in race, no menu)
        simManager.PreventSimulationFinish();
        
        int finishTime = g_lastValidRaceTime;
        log("✓ Race finished at " + finishTime + "ms! Auto-restarting...");
        
        string finishJson = "{\"event\":\"race_finished\",\"finish_time_ms\":" + finishTime + "}";
        SendDataToSocket(finishJson);
        
        g_finishHandled = true;
        g_lastStreamTime = 0;
        
        // Restart race once
        ExecuteCommand("save_replay");
        ExecuteCommand("press delete");
    }
}

void OnLapCountChanged(SimulationManager@ simManager, int count, int target)
{
    string eventJson = "{\"event\":\"lap_completed\",\"lap\":" + count + ",\"total\":" + target + "}";
    SendDataToSocket(eventJson);
}
