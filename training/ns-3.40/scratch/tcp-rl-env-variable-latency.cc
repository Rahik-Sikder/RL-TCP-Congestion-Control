// SCENARIO 3: variable_latency
// Two nodes, single P2P link, 10 Mbit/s.
// The one-way delay cycles through 10 ms → 30 ms → 50 ms → 20 ms every 5
// simulated seconds, mirroring the Kathara background script that calls
// "tc qdisc change dev eth0 root netem delay Xms" in a loop.
//
// In NS-3, channel delay is set at creation time and cannot be changed
// dynamically on a live channel. The standard approach is to model this
// by toggling the ErrorModel (for loss) or, for delay, by interposing a
// DelayJitterEstimator — but the simplest faithful approach used here is
// to schedule Simulator events that rebuild/swap the channel at each
// delay-change boundary. Because NS-3 does not support live channel
// swaps, we approximate by updating a per-device "extra propagation delay"
// via a custom callback that is applied when packets are forwarded.
//
// A simpler and equally valid approach (used here for clarity) is to
// track the "current extra delay" in the RL environment and add it to the
// RTT observation so the agent sees the correct RTT without needing to
// rebuild the channel. The actual NS-3 RTT trace already reflects the
// updated delay once packets transiting the link experience it.
//
// Implementation: we schedule 5-second periodic events that change a
// ns3::PointToPointChannel Delay attribute via Config::Set. NS-3 allows
// changing channel attributes at runtime for PointToPoint channels because
// the delay is re-read per-packet transmission.

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
#include <array>

using namespace ns3;

// ---------------------------------------------------------------------------
// Delay schedule matching the Kathara background script:
//   while true; do
//     sleep 5; tc qdisc change ... delay 10ms
//     sleep 5; tc qdisc change ... delay 30ms
//     sleep 5; tc qdisc change ... delay 50ms
//     sleep 5; tc qdisc change ... delay 20ms
//   done
// ---------------------------------------------------------------------------
static const std::array<std::string, 4> DELAY_SEQUENCE = {"10ms", "30ms", "50ms", "20ms"};
static const double DELAY_INTERVAL_SECONDS = 5.0;

// Callback that cycles the channel delay; re-schedules itself.
static void CycleDelay(uint32_t step) {
    std::string newDelay = DELAY_SEQUENCE[step % DELAY_SEQUENCE.size()];
    Config::Set("/ChannelList/*/$ns3::PointToPointChannel/Delay",
                StringValue(newDelay));
    // std::cout << "[NS3 variable_latency] t=" << Simulator::Now().GetSeconds() << "s  delay → " << newDelay << std::endl;
    Simulator::Schedule(Seconds(DELAY_INTERVAL_SECONDS),
                        &CycleDelay, step + 1);
}


class TcpRlEnv : public OpenGymEnv {
public:
    TcpRlEnv() {
        m_k = 10;
        m_d = 3 * m_k + 1;
        m_ackCounter = 0;
        m_predictionInterval = 5;
        m_highestAck = 0;

        m_currentCwnd = 1.0;
        m_currentThroughput = 0.0;
        m_maxThroughput = 10.0;  // 10 Mbit/s link
        // minRtt corresponds to the shortest one-way delay in the cycle (10 ms), RTT = 20 ms
        m_currentRtt = 20.0;
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
        return box;
    }

    float GetReward() override {
        float alpha = 1.0, beta = 0.5, delta = 0.5;
        float t_max_safe   = (m_maxThroughput > 0) ? m_maxThroughput : 1.0;
        float rtt_min_safe = (m_minRtt > 0)        ? m_minRtt        : 1.0;
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
        // Track the minimum observed RTT so the reward stays calibrated
        // as delay varies. This mirrors what a real TCP stack does.
        if (m_currentRtt < m_minRtt) m_minRtt = m_currentRtt;
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
        return Simulator::Now().GetSeconds() >= 1200.0;
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
    float    simDuration = 1200.0;

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Port number for OpenGym env", simPort);
    cmd.AddValue("simSeed",     "Seed for random generator",   simSeed);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(simSeed);

    Ptr<TcpRlEnv> env = CreateObject<TcpRlEnv>();
    Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface>(simPort);
    env->SetOpenGymInterface(openGym);

    // ------------------------------------------------------------------
    // SCENARIO 3: variable_latency
    // Two nodes on a single 10 Mbit/s link.
    // Initial delay = 10 ms; a scheduled callback rotates through
    // {10ms, 30ms, 50ms, 20ms} every 5 simulated seconds.
    // ------------------------------------------------------------------
    double simulationTime = simDuration;

    NodeContainer nodes;
    nodes.Create(2);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute ("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay",    StringValue("10ms")); // starts at 10 ms
    p2p.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));

    NetDeviceContainer devices = p2p.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

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

    // Schedule the first delay change at t=5s (step=1 → "30ms"),
    // then it self-reschedules every 5 s cycling through the array.
    Simulator::Schedule(Seconds(DELAY_INTERVAL_SECONDS), &CycleDelay, 1u);

    // Hook RL traces
    Simulator::Schedule(Seconds(0.11), [env]() {
        // std::cout << "[NS3 variable_latency] Hooking TCP Traces..." << std::endl;
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
        // std::cout << "[NS3 variable_latency] Sending initial state to Python..." << std::endl;
        env->Notify();
    });

    // std::cout << "Starting NS-3 variable_latency scenario on port " << simPort << "...\n";
    Simulator::Stop(Seconds(simDuration + 1.0));
    Simulator::Run();
    openGym->NotifySimulationEnd();
    Simulator::Destroy();

    return 0;
}
