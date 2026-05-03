// SCENARIO 2: bottleneck
// Three nodes: sender -- router -- receiver
//   sender ↔ router:   10 Mbit/s, 5 ms  (link A)
//   router ↔ receiver:  1 Mbit/s, 5 ms  (link B — bottleneck)
//
// The bottleneck link has a small queue (10 packets) to force congestion,
// mirroring the Kathara tbf qdisc (rate 1mbit burst 10kb latency 50ms).
// RTT min ≈ 2*(5+5) = 20 ms; T_max is capped by the 1 Mbit bottleneck.

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/opengym-module.h"
#include <deque>
#include <string>
#include <cmath>

using namespace ns3;

class TcpRlEnv : public OpenGymEnv {
public:
    TcpRlEnv() {
        m_k = 10;
        m_d = 3 * m_k + 1 + 1;
        m_ackCounter = 0;
        m_predictionInterval = 5;
        m_highestAck = 0;

        m_currentCwnd = 1.0;
        m_currentThroughput = 0.0;
        m_maxThroughput = 1.0;   // Bottleneck is 1 Mbit/s
        m_currentRtt = 20.0;     // ~2*(5ms+5ms)
        m_minRtt = 20.0;
        m_lossRate = 0.0;
        m_packetsLost = 0;
        m_packetsSent = 0;
    }

    Ptr<OpenGymSpace> GetObservationSpace() override {
        return CreateObject<OpenGymBoxSpace>(-1e38f, 1e38f, std::vector<uint32_t>{m_d}, "float32");
    }

    Ptr<OpenGymSpace> GetActionSpace() override {
        return CreateObject<OpenGymBoxSpace>(-1.0f, 1.0f, std::vector<uint32_t>{1}, "float32");
    }

    Ptr<OpenGymDataContainer> GetObservation() override {
        auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{m_d});
        auto padAndAdd = [&](const std::deque<float>& history) {
            int padding = m_k - history.size();
            for (int i = 0; i < padding; ++i) box->AddValue(0.0f);
            for (float val : history) box->AddValue(val);
        };
        padAndAdd(m_rttHistory);
        padAndAdd(m_dupAckHistory);
        padAndAdd(m_timeoutHistory);
        box->AddValue(m_currentCwnd);
        box->AddValue(m_lossRate);
        return box;
    }

float GetReward() override {
        // R = a(T/T_max) - b(RTT/RTT_min) - d(loss_rate) - g(timeout)
        float alpha = 4.0, beta = 0.5, delta = 2.0; 
        float gamma = 5.0; // New heavy penalty weight for timeouts
        
        // Safety checks to prevent division by zero
        float t_max_safe = (m_maxThroughput > 0) ? m_maxThroughput : 1.0;
        float rtt_min_safe = (m_minRtt > 0) ? m_minRtt : 1.0;

        float t_term = alpha * (m_currentThroughput / t_max_safe);
        float rtt_term = beta * (m_currentRtt / rtt_min_safe);
        float loss_term = delta * m_lossRate;
        
        // Check if the most recent network event recorded was a timeout
        float timeout_term = 0.0f;
        if (!m_timeoutHistory.empty() && m_timeoutHistory.back() == 1.0f) {
            timeout_term = gamma * 1.0f; 
        }
        
        return t_term - rtt_term - loss_term - timeout_term;
    }

    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override {
        Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
        float a_agent = box->GetValue(0);
        m_currentCwnd = m_currentCwnd * pow(2.0, a_agent);
        float maxCwnd = GetBandwidthCwndCapMss();
        if (m_currentCwnd > maxCwnd) m_currentCwnd = maxCwnd;
        if (m_currentCwnd < 1.0) m_currentCwnd = 1.0;
        if (m_tcpSocket) {
            uint32_t segmentSize = 1448;
            m_tcpSocket->SetAttribute("SndCwnd",
                UintegerValue(static_cast<uint32_t>(m_currentCwnd * segmentSize)));
        }
        return true;
    }

    std::string GetExtraInfo() override {
        std::string info = "{";
        info += "\"throughput_mbps\": "  + std::to_string(m_currentThroughput) + ",";
        info += "\"avg_rtt_ms\": "       + std::to_string(m_currentRtt)        + ",";
        info += "\"packet_loss_rate\": " + std::to_string(m_lossRate);
        info += "}";
        return info;
    }

    void AttachTcpSocket(Ptr<TcpSocketBase> socket) { m_tcpSocket = socket; }

    void OnPacketSent(Ptr<const Packet> packet, const TcpHeader& header,
                      Ptr<const TcpSocketBase> socket) {
        m_packetsSent++;
    }

    void OnPacketDropped(Ptr<const Packet> packet) {
        m_packetsLost++;
        if (m_packetsSent > 0)
            m_lossRate = static_cast<float>(m_packetsLost) / m_packetsSent;
    }

    void OnRttUpdated(Time oldRtt, Time newRtt) {
        m_currentRtt = newRtt.GetMilliSeconds();
    }

    void OnCongStateChange(const TcpSocketState::TcpCongState_t oldState,
                           const TcpSocketState::TcpCongState_t newState) {
        (void)oldState;
        if (newState == TcpSocketState::CA_LOSS) {
            m_timeoutHistory.push_back(1.0f);
            if (m_timeoutHistory.size() > m_k) m_timeoutHistory.pop_front();
            CalculateThroughput();
            Notify();
        }
    }

    void OnAckReceived(Ptr<const Packet> packet, const TcpHeader& header,
                       Ptr<const TcpSocketBase> socket) {
        bool isDupAck = false;
        if (header.GetFlags() & TcpHeader::ACK) {
            if (header.GetAckNumber() == m_highestAck) {
                isDupAck = true;
            } else if (header.GetAckNumber() > m_highestAck) {
                m_highestAck = header.GetAckNumber();
            }
        }
        bool isTimeout = false;

        m_rttHistory.push_back(m_currentRtt);
        if (m_rttHistory.size() > m_k) m_rttHistory.pop_front();

        m_dupAckHistory.push_back(isDupAck ? 1.0f : 0.0f);
        if (m_dupAckHistory.size() > m_k) m_dupAckHistory.pop_front();

        m_timeoutHistory.push_back(isTimeout ? 1.0f : 0.0f);
        if (m_timeoutHistory.size() > m_k) m_timeoutHistory.pop_front();

        m_ackCounter++;
        if (m_ackCounter % m_predictionInterval == 0) {
            CalculateThroughput();
            Notify();
        }
    }

    void CalculateThroughput() {
        if (m_currentRtt > 0) {
            float bytesInFlight = m_currentCwnd * 1448.0f;
            float rttSeconds    = m_currentRtt / 1000.0f;
            m_currentThroughput = (bytesInFlight * 8.0f / rttSeconds) / 1000000.0f;
        }
    }

    bool GetGameOver() override {
        return Simulator::Now().GetSeconds() >= 300.0;
    }

private:
    float GetBandwidthCwndCapMss() const {
        return 1000000.0f; // Return a massive hard limit (1 million MSS)
    }

    uint32_t m_k, m_d;
    uint32_t m_ackCounter;
    uint32_t m_predictionInterval;

    std::deque<float> m_rttHistory, m_dupAckHistory, m_timeoutHistory;
    float m_currentCwnd, m_currentThroughput, m_maxThroughput;
    float m_currentRtt, m_minRtt, m_lossRate;
    uint32_t m_packetsSent, m_packetsLost;

    SequenceNumber32 m_highestAck;
    Ptr<TcpSocketBase> m_tcpSocket;
};


int main(int argc, char *argv[]) {
    uint16_t simPort = 5555;
    uint32_t simSeed = 42;
    float    simDuration = 300.0;

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port number for OpenGym env", simPort);
    cmd.AddValue("simSeed",     "Seed for random generator",   simSeed);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(simSeed);

    Ptr<TcpRlEnv> env = CreateObject<TcpRlEnv>();
    Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(simPort);
    env->SetOpenGymInterface(openGym);

    // ------------------------------------------------------------------
    // SCENARIO 2: bottleneck
    // Nodes: sender (0) -- router (1) -- receiver (2)
    //
    // Link A (sender ↔ router):   10 Mbit/s, 5 ms, large queue
    // Link B (router ↔ receiver):  1 Mbit/s, 5 ms, small queue (10p)
    //   → forces congestion at the router egress, matching the Kathara
    //     tbf token-bucket filter on the router's eth1.
    //
    // IP plan:
    //   Link A: 10.0.0.0/24  sender=10.0.0.1  router=10.0.0.254
    //   Link B: 10.0.1.0/24  router=10.0.1.254  receiver=10.0.1.2
    // ------------------------------------------------------------------
    double simulationTime = simDuration;

    NodeContainer allNodes;
    allNodes.Create(3);
    Ptr<Node> sender   = allNodes.Get(0);
    Ptr<Node> router   = allNodes.Get(1);
    Ptr<Node> receiver = allNodes.Get(2);

    // Link A: sender ↔ router (10 Mbit/s, 5 ms)
    PointToPointHelper p2pA;
    p2pA.SetDeviceAttribute ("DataRate", StringValue("10Mbps"));
    p2pA.SetChannelAttribute("Delay",    StringValue("5ms"));
    p2pA.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));
    NodeContainer pairA(sender, router);
    NetDeviceContainer devicesA = p2pA.Install(pairA);

    // Link B: router ↔ receiver (1 Mbit/s bottleneck, 5 ms, 10-packet queue)
    PointToPointHelper p2pB;
    p2pB.SetDeviceAttribute ("DataRate", StringValue("1Mbps"));
    p2pB.SetChannelAttribute("Delay",    StringValue("5ms"));
    p2pB.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("10p")); // tight queue = congestion
    NodeContainer pairB(router, receiver);
    NetDeviceContainer devicesB = p2pB.Install(pairB);

    InternetStackHelper stack;
    stack.Install(allNodes);

    // Link A addresses
    Ipv4AddressHelper addrA;
    addrA.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer ifA = addrA.Assign(devicesA);
    // sender=10.0.0.1, router(eth0)=10.0.0.254 — note: NS-3 assigns .1 and .2;
    // addressing matches intent (sender on A, receiver on B).

    // Link B addresses
    Ipv4AddressHelper addrB;
    addrB.SetBase("10.0.1.0", "255.255.255.0");
    Ipv4InterfaceContainer ifB = addrB.Assign(devicesB);
    // router(eth1)=10.0.1.1, receiver=10.0.1.2

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // Receiver application (sink on receiver node)
    uint16_t sinkPort = 50000;
    Address sinkAddress(InetSocketAddress(ifB.GetAddress(1), sinkPort));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", sinkAddress);
    ApplicationContainer sinkApps = packetSinkHelper.Install(receiver);
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(simulationTime));

    // Sender application (bulk send from sender node)
    BulkSendHelper source("ns3::TcpSocketFactory", sinkAddress);
    source.SetAttribute("MaxBytes", UintegerValue(0));
    ApplicationContainer sourceApps = source.Install(sender);
    sourceApps.Start(Seconds(0.1));
    sourceApps.Stop(Seconds(simulationTime));

    // Hook RL traces on the sender node (NodeList/0)
    Simulator::Schedule(Seconds(0.11), [env]() {
        // std::cout << "[NS3 bottleneck] Hooking TCP Traces..." << std::endl;
        Config::ConnectWithoutContext(
            "/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/Tx",
            MakeCallback(&TcpRlEnv::OnPacketSent, env));
        Config::ConnectWithoutContext(
            "/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/Rx",
            MakeCallback(&TcpRlEnv::OnAckReceived, env));
        Config::ConnectWithoutContext(
            "/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/RTT",
            MakeCallback(&TcpRlEnv::OnRttUpdated, env));
        Config::ConnectWithoutContext(
            "/NodeList/0/$ns3::TcpL4Protocol/SocketList/*/CongState",
            MakeCallback(&TcpRlEnv::OnCongStateChange, env));
        // Drop trace on sender's egress device (link A)
        Config::ConnectWithoutContext(
            "/NodeList/0/DeviceList/0/$ns3::PointToPointNetDevice/TxQueue/Drop",
            MakeCallback(&TcpRlEnv::OnPacketDropped, env));
        // Also catch drops at the bottleneck queue (router's link B device = NodeList/1/DeviceList/1)
        Config::ConnectWithoutContext(
            "/NodeList/1/DeviceList/1/$ns3::PointToPointNetDevice/TxQueue/Drop",
            MakeCallback(&TcpRlEnv::OnPacketDropped, env));
    });

    Simulator::Schedule(Seconds(0.12), [env]() {
        // std::cout << "[NS3 bottleneck] Sending initial state to Python..." << std::endl;
        env->Notify();
    });

    // std::cout << "Starting NS-3 bottleneck scenario on port " << simPort << "...\n";
    Simulator::Stop(Seconds(simDuration + 1.0));
    Simulator::Run();
    openGym->NotifySimulationEnd();
    Simulator::Destroy();

    return 0;
}
