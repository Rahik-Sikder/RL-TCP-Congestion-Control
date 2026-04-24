#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/opengym-module.h"
#include <deque>
#include <string>
#include <cmath>
#include <algorithm>

using namespace ns3;

class TcpRlCongestionOps : public TcpCongestionOps {
public:
    static TypeId GetTypeId() {
        static TypeId tid = TypeId("ns3::TcpRlCongestionOps")
            .SetParent<TcpCongestionOps>()
            .SetGroupName("Internet")
            .AddConstructor<TcpRlCongestionOps>();
        return tid;
    }

    TcpRlCongestionOps() : TcpCongestionOps(), m_targetCwndSegments(1.0) {}
    TcpRlCongestionOps(const TcpRlCongestionOps& sock)
        : TcpCongestionOps(sock), m_targetCwndSegments(sock.m_targetCwndSegments) {}
    ~TcpRlCongestionOps() override = default;

    void SetTargetCwndSegments(double cwndSegments) {
        m_targetCwndSegments = std::max(1.0, cwndSegments);
    }

    std::string GetName() const override {
        return "TcpRlCongestionOps";
    }

    uint32_t GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight) override {
        (void)bytesInFlight;
        return std::max(2 * tcb->m_segmentSize, GetTargetCwndBytes(tcb));
    }

    void IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked) override {
        (void)segmentsAcked;
        ApplyTargetCwnd(tcb);
    }

    void PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt) override {
        (void)segmentsAcked;
        (void)rtt;
        ApplyTargetCwnd(tcb);
    }

    void CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event) override {
        (void)event;
        ApplyTargetCwnd(tcb);
    }

    void CongestionStateSet(Ptr<TcpSocketState> tcb,
                            const TcpSocketState::TcpCongState_t newState) override {
        (void)newState;
        ApplyTargetCwnd(tcb);
    }

    bool HasCongControl() const override {
        return true;
    }

    void CongControl(Ptr<TcpSocketState> tcb,
                     const TcpRateOps::TcpRateConnection& rc,
                     const TcpRateOps::TcpRateSample& rs) override {
        (void)rc;
        (void)rs;
        ApplyTargetCwnd(tcb);
    }

    Ptr<TcpCongestionOps> Fork() override {
        return CopyObject<TcpRlCongestionOps>(this);
    }

private:
    uint32_t GetTargetCwndBytes(Ptr<const TcpSocketState> tcb) const {
        return static_cast<uint32_t>(std::llround(m_targetCwndSegments * tcb->m_segmentSize));
    }

    void ApplyTargetCwnd(Ptr<TcpSocketState> tcb) const {
        const uint32_t targetBytes = std::max(tcb->m_segmentSize, GetTargetCwndBytes(tcb));
        tcb->m_cWnd = targetBytes;
        tcb->m_cWndInfl = targetBytes;
        tcb->m_ssThresh = std::max(2 * tcb->m_segmentSize, targetBytes);
    }

    double m_targetCwndSegments;
};

class TcpRlEnv : public OpenGymEnv {
public:
    TcpRlEnv() {
        m_k = 10; // Window size [cite: 38]
        m_d = 3 * m_k + 1; // 31 features total [cite: 39]
        m_ackCounter = 0;          
        m_predictionInterval = 5;  // Call agent every 5 ACKs
        m_highestAck = 0; // NEW: Initialize sequence number

        // Initializing default metric values
        m_currentCwnd = 1.0; // Assume MSS multiplier
        m_currentThroughput = 0.0;
        m_maxThroughput = 10.0; // Set to your bottleneck bandwidth
        m_currentRtt = 10.0; 
        m_minRtt = 10.0; // Set to your propagation delay
        m_lossRate = 0.0;
        m_packetsLost = 0;
        m_packetsSent = 0;
    }

    void SetBottleneckRate(const std::string& dataRate)
    {
        DataRate bw(dataRate);
        m_bottleneckBitrate = static_cast<double>(bw.GetBitRate());
    }

    // 1. Define State Space (d = 31 continuous values)
    Ptr<OpenGymSpace> GetObservationSpace() override {
        return CreateObject<OpenGymBoxSpace>(-1e38f, 1e38f, std::vector<uint32_t>{m_d}, "float32");
    }

    // 2. Define Action Space (Continuous [-1, 1] for PPO/DDPG) [cite: 41]
    Ptr<OpenGymSpace> GetActionSpace() override {
        return CreateObject<OpenGymBoxSpace>(-1.0f, 1.0f, std::vector<uint32_t>{1}, "float32");
    }

    // 3. Package the Observation [cite: 38]
    Ptr<OpenGymDataContainer> GetObservation() override {
        auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{m_d});
        
        // Helper lambda to pad deques with 0.0 if we haven't received k packets yet
        auto padAndAdd = [&](const std::deque<float>& history) {
            int padding = m_k - history.size();
            for (int i = 0; i < padding; ++i) box->AddValue(0.0f);
            for (float val : history) box->AddValue(val);
        };

        // Add k RTTs, k dupACKs, k timeouts
        padAndAdd(m_rttHistory);
        padAndAdd(m_dupAckHistory);
        padAndAdd(m_timeoutHistory);
        
        box->AddValue(m_currentCwnd); // Last element is cwnd [cite: 38]

        // Print buffers and PID
        // std::cout << "[tcp-rl-env] PID: " << getpid() << std::endl;
        // std::cout << "  RTT History: ";
        // for (const auto& v : m_rttHistory) std::cout << v << " ";
        // std::cout << std::endl;
        // std::cout << "  DupAck History: ";
        // for (const auto& v : m_dupAckHistory) std::cout << v << " ";
        // std::cout << std::endl;
        // std::cout << "  Timeout History: ";
        // for (const auto& v : m_timeoutHistory) std::cout << v << " ";
        // std::cout << std::endl;

        // // Sleep for 2 seconds
        // sleep(1);
        return box;
    }

    // 4. Calculate Reward [cite: 43]
    float GetReward() override {
        // R = a(T/T_max) - b(RTT/RTT_min) - d(loss_rate) [cite: 44, 45, 46, 47, 48, 49]
        float alpha = 1.0, beta = 0.5, delta = 0.5; // Grid search weights [cite: 50]
        
        // Safety checks to prevent division by zero
        float t_max_safe = (m_maxThroughput > 0) ? m_maxThroughput : 1.0;
        float rtt_min_safe = (m_minRtt > 0) ? m_minRtt : 1.0;

        float t_term = alpha * (m_currentThroughput / t_max_safe);
        float rtt_term = beta * (m_currentRtt / rtt_min_safe);
        float loss_term = delta * m_lossRate;
        
        return t_term - rtt_term - loss_term;
    }

    // 5. Apply the Action from Python
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override {

        
        Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
        float a_agent = box->GetValue(0); // Value between -1 and 1 
        // std::cout << "[ACTION] t=" << Simulator::Now().GetSeconds()
        //           << " a_agent=" << a_agent
        //           << " cwnd_mss_before=" << m_currentCwnd;
        
        // Apply: cwnd_{t+1} = cwnd_t * 2^{a_agent} [cite: 42]
        m_currentCwnd = m_currentCwnd * pow(2.0, a_agent); 
        // Bound cwnd by bandwidth (BDP) instead of a fixed constant.
        float maxCwnd = GetBandwidthCwndCapMss();
        if (m_currentCwnd > maxCwnd) m_currentCwnd = maxCwnd;
        // Ensure cwnd doesn't drop below 1 MSS
        if (m_currentCwnd < 1.0) m_currentCwnd = 1.0;
        std::cout << " cwnd_mss_after=" << m_currentCwnd << std::endl;

        // Push action into congestion control module so next ACK updates real TCP cwnd.
        if (m_rlCongestion) {
            m_rlCongestion->SetTargetCwndSegments(m_currentCwnd);
        }

        return true; 
    }

    // 6. Report Metrics back to Python 
    std::string GetExtraInfo() override {
        std::string info = "{";
        info += "\"throughput_mbps\": " + std::to_string(m_currentThroughput) + ",";
        info += "\"avg_rtt_ms\": " + std::to_string(m_currentRtt) + ",";
        info += "\"packet_loss_rate\": " + std::to_string(m_lossRate);
        info += "}";
        return info;
    }

    // ---------------------------------------------------------------------
    // NS-3 SIMULATION HOOKS & LOGIC
    // ---------------------------------------------------------------------

    // Helper method to attach a specific sender socket to this environment
    void AttachTcpSocket(Ptr<TcpSocketBase> socket) {
        if (m_tcpSocket) {
            return;
        }
        m_tcpSocket = socket;
        m_rlCongestion = CreateObject<TcpRlCongestionOps>();
        m_rlCongestion->SetTargetCwndSegments(m_currentCwnd);
        m_tcpSocket->SetCongestionControlAlgorithm(m_rlCongestion);
        UintegerValue segSizeAttr;
        m_tcpSocket->GetAttribute("SegmentSize", segSizeAttr);
        m_segmentSizeBytes = std::max<uint32_t>(1u, static_cast<uint32_t>(segSizeAttr.Get()));
        std::cout << "[NS3] Attached sender socket and installed TcpRlCongestionOps" << std::endl;
    }

    void AttachTcpSocketFromTrace(Ptr<const TcpSocketBase> socket) {
        if (!m_tcpSocket && socket) {
            AttachTcpSocket(ConstCast<TcpSocketBase>(socket));
        }
    }

    // Matches TcpSocketBase "Tx" trace: (Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
    void OnPacketSent(Ptr<const Packet> packet, const TcpHeader& header, Ptr<const TcpSocketBase> socket) {
        AttachTcpSocketFromTrace(socket);
        m_packetsSent++;
    }

    // Matches DropTailQueue "Drop" trace: (Ptr<const Packet>)
    void OnPacketDropped(Ptr<const Packet> packet) {
        m_packetsLost++;
        // Update loss rate metric for the reward calculation
        if (m_packetsSent > 0) {
            m_lossRate = static_cast<float>(m_packetsLost) / m_packetsSent;
        }
    }

// NEW: Dedicated hook for RTT updates
    void OnRttUpdated(Time oldRtt, Time newRtt) {
        m_currentRtt = newRtt.GetMilliSeconds();
    }

    // Validate that NS-3 TCP stack itself is changing cwnd over time (bytes).
    void OnCwndChange(uint32_t oldCwnd, uint32_t newCwnd) {
        if (m_tcpSocket) {
            m_currentCwnd = static_cast<float>(newCwnd) / static_cast<float>(m_segmentSizeBytes);
        }
    //     std::cout << "[CWND] t=" << Simulator::Now().GetSeconds()
    //               << " old_bytes=" << oldCwnd
    //               << " new_bytes=" << newCwnd << std::endl;
    }

    // UPDATED: Correct NS-3 Rx trace signature
    void OnAckReceived(Ptr<const Packet> packet, const TcpHeader& header, Ptr<const TcpSocketBase> socket) {
        AttachTcpSocketFromTrace(socket);
        // 1. Calculate if this is a Duplicate ACK
        bool isDupAck = false;
        if (header.GetFlags() & TcpHeader::ACK) {
            if (header.GetAckNumber() == m_highestAck) {
                isDupAck = true;
            } else if (header.GetAckNumber() > m_highestAck) {
                m_highestAck = header.GetAckNumber();
            }
        }

        // 2. Timeouts are handled internally by NS-3. For this Rx loop, 
        // receiving a packet implies no timeout occurred on this step.
        bool isTimeout = false; 

        // 3. Update sliding buffers
        m_rttHistory.push_back(m_currentRtt);
        if (m_rttHistory.size() > m_k) m_rttHistory.pop_front();

        m_dupAckHistory.push_back(isDupAck ? 1.0f : 0.0f);
        if (m_dupAckHistory.size() > m_k) m_dupAckHistory.pop_front();

        m_timeoutHistory.push_back(isTimeout ? 1.0f : 0.0f);
        if (m_timeoutHistory.size() > m_k) m_timeoutHistory.pop_front();

        // 4. Increment the ACK counter
        m_ackCounter++;

        // 5. Trigger the Python Agent every 5 ACKs
        if (m_ackCounter % m_predictionInterval == 0) {
            // std::cout << "\n[NS3] SUCCESS: Sent ACK received! Network traffic is flowing.\n" << std::endl;

            CalculateThroughput(); 
            Notify(); 
        }
    }

    // Helper to calculate throughput (Needs access to a sink application to be perfectly accurate, 
    // but can be estimated here based on ACKs and RTT)
    void CalculateThroughput() {
        // Simple heuristic: (MSS * m_currentCwnd) / RTT
        // For accurate throughput, connect this class to the sink's Rx packet traces.
        if (m_currentRtt > 0) {
            float bytesInFlight = m_currentCwnd * static_cast<float>(m_segmentSizeBytes); // in bytes
            float rttSeconds = m_currentRtt / 1000.0f;
            m_currentThroughput = (bytesInFlight * 8.0f / rttSeconds) / 1000000.0f; // Mbps
        }
    }
    
    bool GetGameOver() override {
        // Example: End the episode after 1 hour of simulated time
        return Simulator::Now().GetSeconds() >= 1 * 3600;
    }

private:
    float GetBandwidthCwndCapMss() const
    {
        if (m_bottleneckBitrate <= 0.0)
        {
            return 65535.0f;
        }
        const float rttMs = (m_minRtt > 0.0f) ? m_minRtt : 10.0f;
        const float rttSeconds = rttMs / 1000.0f;
        const float bdpBytes = static_cast<float>((m_bottleneckBitrate * rttSeconds) / 8.0);
        const float capMss = bdpBytes / static_cast<float>(m_segmentSizeBytes);
        return std::max(1.0f, capMss);
    }

    uint32_t m_k, m_d;
    uint32_t m_ackCounter;
    uint32_t m_predictionInterval;

    std::deque<float> m_rttHistory, m_dupAckHistory, m_timeoutHistory;
    float m_currentCwnd, m_currentThroughput, m_maxThroughput, m_currentRtt, m_minRtt, m_lossRate;

    uint32_t m_packetsSent;
    uint32_t m_packetsLost;

    SequenceNumber32 m_highestAck;
    Ptr<TcpSocketBase> m_tcpSocket;
    Ptr<TcpRlCongestionOps> m_rlCongestion;
    uint32_t m_segmentSizeBytes{1448};
    double m_bottleneckBitrate{0.0};
};


int main(int argc, char *argv[]) {
    // Default parameters that Python can override
    uint16_t simPort = 5555;
    uint32_t simSeed = 42;
    float simDuration = 60.0; // Seconds

    // Allow Python/ns3-gym to pass arguments in
    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port number for OpenGym env", simPort);
    cmd.AddValue("simSeed", "Seed for random generator", simSeed);
    cmd.Parse(argc, argv);

    // Seed the random number generator
    RngSeedManager::SetSeed(simSeed);

    // 1. Initialize your custom RL Environment
    Ptr<TcpRlEnv> env = CreateObject<TcpRlEnv>();

    // 2. Bind the environment to the ns3-gym Python Interface
    Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(simPort);
    env->SetOpenGymInterface(openGym);
    // openGym->SetGetActionSpaceCb(MakeCallback(&TcpRlEnv::GetActionSpace, env));
    // openGym->SetGetObservationSpaceCb(MakeCallback(&TcpRlEnv::GetObservationSpace, env));
    // openGym->SetGetGameOverCb(MakeCallback(&TcpRlEnv::GetGameOver, env));
    // openGym->SetGetObservationCb(MakeCallback(&TcpRlEnv::GetObservation, env));
    // openGym->SetGetRewardCb(MakeCallback(&TcpRlEnv::GetReward, env));
    // openGym->SetGetExtraInfoCb(MakeCallback(&TcpRlEnv::GetExtraInfo, env));
    // openGym->SetExecuteActionsCb(MakeCallback(&TcpRlEnv::ExecuteActions, env));

    // ------------------------------------------------------------------
    // 3. BUILD YOUR NETWORK TOPOLOGY HERE
// Configurable network parameters
    std::string dataRate = "5Mbps";
    std::string delay = "10ms";
    double simulationTime = simDuration; 
    env->SetBottleneckRate(dataRate);

    // Create the nodes
    NodeContainer nodes;
    nodes.Create(2); // Node 0 is Sender, Node 1 is Receiver

    // Create the Point-to-Point link
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(dataRate));
    p2p.SetChannelAttribute("Delay", StringValue(delay));
    
    // Important for buffer bloat/congestion scenarios: Set the queue size
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("20p"));

    // Install link devices onto the nodes
    NetDeviceContainer devices = p2p.Install(nodes);
    // Add a very small random loss rate on the receiver-side device.
    Ptr<RateErrorModel> smallLoss = CreateObject<RateErrorModel>();
    smallLoss->SetAttribute("ErrorRate", DoubleValue(1e-6));
    devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(smallLoss));

    // Install the Internet Stack (TCP/IP)
    InternetStackHelper stack;
    stack.Install(nodes);

    // Assign IP Addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Set up the Receiver Application (Packet Sink)
    uint16_t sinkPort = 50000;
    Address sinkAddress(InetSocketAddress(interfaces.GetAddress(1), sinkPort));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", sinkAddress);
    ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(1));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(simulationTime));

    // Set up the Sender Application (Bulk Send)
    // This constantly tries to send data, which will trigger congestion control
    BulkSendHelper source("ns3::TcpSocketFactory", sinkAddress);
    source.SetAttribute("MaxBytes", UintegerValue(0)); // 0 means send infinitely
    ApplicationContainer sourceApps = source.Install(nodes.Get(0));
    sourceApps.Start(Seconds(0.1)); // Start slightly after receiver
    sourceApps.Stop(Seconds(simulationTime));

    // Populate routing tables
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ------------------------------------------------------------------
    // CONNECTING THE RL ENVIRONMENT TO THE TOPOLOGY
    // ------------------------------------------------------------------
    // NS-3 creates sockets dynamically when the application starts (at t=0.1s).
    // Schedule trace connections slightly after app start so sockets exist.
    Simulator::Schedule(Seconds(0.11), [env]() {
        std::cout << "[NS3] Hooking TCP Traces..." << std::endl;
        
        Config::ConnectWithoutContext("/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/Tx",
            MakeCallback(&TcpRlEnv::OnPacketSent, env));
        Config::ConnectWithoutContext("/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/Rx",
            MakeCallback(&TcpRlEnv::OnAckReceived, env));
        Config::ConnectWithoutContext("/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/RTT",
            MakeCallback(&TcpRlEnv::OnRttUpdated, env));
        Config::ConnectWithoutContext("/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/CongestionWindow",
            MakeCallback(&TcpRlEnv::OnCwndChange, env));

        Config::ConnectWithoutContext("/NodeList/0/DeviceList/0/$ns3::PointToPointNetDevice/TxQueue/Drop",
            MakeCallback(&TcpRlEnv::OnPacketDropped, env));
    });

    Simulator::Schedule(Seconds(0.12), [env]() {
        std::cout << "[NS3] Sending initial state to Python..." << std::endl;
        env->Notify(); 
    });
    // ------------------------------------------------------------------

    // 4. Run the Simulation Loop
    std::cout << "Starting NS-3 Simulation for TCP RL on port " << simPort << "...\n";
    
    // Stop the simulation slightly after the RL episode ends
    Simulator::Stop(Seconds(simDuration + 1.0)); 
    
    Simulator::Run();
    openGym->NotifySimulationEnd();
    
    Simulator::Destroy();

    return 0;
}
