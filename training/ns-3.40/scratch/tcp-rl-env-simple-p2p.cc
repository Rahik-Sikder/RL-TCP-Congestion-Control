// SCENARIO 1: simple_p2p
// Two nodes on a single link: 10Mbit, 20ms delay, no loss.
// Mirrors the Kathara lab where sender has 10.0.0.1/24 and receiver has 10.0.0.2/24
// on the same segment, with a netem qdisc shaping the sender egress.

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
        m_d = 3 * m_k + 1; // 31 features
        m_ackCounter = 0;
        m_predictionInterval = 5;
        m_highestAck = 0;

        m_currentCwnd = 1.0;
        m_currentThroughput = 0.0;
        m_maxThroughput = 10.0;  // 10 Mbit/s link
        m_currentRtt = 40.0;     // ~2 * 20ms propagation
        m_minRtt = 40.0;
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
        return box;
    }

    float GetReward() override {
        float alpha = 1.0, beta = 0.5, delta = 0.5;
        float t_max_safe = (m_maxThroughput > 0) ? m_maxThroughput : 1.0;
        float rtt_min_safe = (m_minRtt > 0) ? m_minRtt : 1.0;
        return alpha * (m_currentThroughput / t_max_safe)
             - beta  * (m_currentRtt / rtt_min_safe)
             - delta * m_lossRate;
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
        info += "\"throughput_mbps\": " + std::to_string(m_currentThroughput) + ",";
        info += "\"avg_rtt_ms\": "      + std::to_string(m_currentRtt)        + ",";
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
            float rttSeconds = m_currentRtt / 1000.0f;
            m_currentThroughput = (bytesInFlight * 8.0f / rttSeconds) / 1000000.0f;
        }
    }

    bool GetGameOver() override {
        return Simulator::Now().GetSeconds() >= 120.0;
    }

private:
    float GetBandwidthCwndCapMss() const {
        const float throughputMbps = (m_maxThroughput > 0.0f) ? m_maxThroughput : 1.0f;
        const float rttMs = (m_minRtt > 0.0f) ? m_minRtt : 1.0f;
        const float bdpBytes = (throughputMbps * 1000000.0f) * (rttMs / 1000.0f) / 8.0f;
        const float capMss = bdpBytes / 1448.0f;
        return (capMss > 1.0f) ? capMss : 1.0f;
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
    float    simDuration = 120.0;

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port number for OpenGym env", simPort);
    cmd.AddValue("simSeed",     "Seed for random generator",   simSeed);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(simSeed);

    Ptr<TcpRlEnv> env = CreateObject<TcpRlEnv>();
    Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(simPort);
    env->SetOpenGymInterface(openGym);

    // ------------------------------------------------------------------
    // SCENARIO 1: simple_p2p
    // Single P2P link: 10 Mbit/s, 20 ms delay (RTT ~40 ms), no loss.
    // Matches Kathara netem: delay 20ms rate 10mbit on sender egress.
    // ------------------------------------------------------------------
    std::string dataRate  = "10Mbps";
    std::string delay     = "20ms";   // one-way; RTT ≈ 40 ms
    double simulationTime = simDuration;

    NodeContainer nodes;
    nodes.Create(2); // Node 0 = sender (10.0.0.1), Node 1 = receiver (10.0.0.2)

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute ("DataRate", StringValue(dataRate));
    p2p.SetChannelAttribute("Delay",    StringValue(delay));
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));

    NetDeviceContainer devices = p2p.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
    // Node 0 → 10.0.0.1, Node 1 → 10.0.0.2  (matches Kathara addressing)

    uint16_t sinkPort = 50000;
    Address sinkAddress(InetSocketAddress(interfaces.GetAddress(1), sinkPort));

    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", sinkAddress);
    ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(1));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(simulationTime));

    BulkSendHelper source("ns3::TcpSocketFactory", sinkAddress);
    source.SetAttribute("MaxBytes", UintegerValue(0));
    ApplicationContainer sourceApps = source.Install(nodes.Get(0));
    sourceApps.Start(Seconds(0.1));
    sourceApps.Stop(Seconds(simulationTime));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    Simulator::Schedule(Seconds(0.11), [env]() {
        // std::cout << "[NS3 simple_p2p] Hooking TCP Traces..." << std::endl;
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
            "/NodeList/0/DeviceList/0/$ns3::PointToPointNetDevice/TxQueue/Drop",
            MakeCallback(&TcpRlEnv::OnPacketDropped, env));
    });

    Simulator::Schedule(Seconds(0.12), [env]() {
        // std::cout << "[NS3 simple_p2p] Sending initial state to Python..." << std::endl;
        env->Notify();
    });

    // std::cout << "Starting NS-3 simple_p2p scenario on port " << simPort << "...\n";
    Simulator::Stop(Seconds(simDuration + 1.0));
    Simulator::Run();
    openGym->NotifySimulationEnd();
    Simulator::Destroy();

    return 0;
}
