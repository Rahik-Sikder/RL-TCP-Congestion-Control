#include <arpa/inet.h>
#include <getopt.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "transport/transport.h"

#ifdef DEBUG
#define DBG(...) fprintf(stderr, "[receiver] " __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

static volatile int g_running = 1;
static uint64_t     g_total_bytes = 0;

static void handle_sigint(int sig) {
    (void)sig;
    g_running = 0;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --port PORT\n", prog);
    exit(1);
}

/* Buffer large enough for header + one MSS payload */
#define RECV_BUF (sizeof(pkt_hdr_t) + MSS + 64)

int main(int argc, char **argv) {
    int port = 5000;

    static struct option long_opts[] = {
        {"port", required_argument, 0, 'p'},
        {0, 0, 0, 0},
    };

    int ch;
    while ((ch = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (ch) {
        case 'p': port = atoi(optarg); break;
        default:  usage(argv[0]);
        }
    }

    signal(SIGINT, handle_sigint);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    /* Reuse port so restarts are quick */
    int yes = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in bind_addr;
    memset(&bind_addr, 0, sizeof(bind_addr));
    bind_addr.sin_family      = AF_INET;
    bind_addr.sin_port        = htons((uint16_t)port);
    bind_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr *)&bind_addr, sizeof(bind_addr)) < 0) {
        perror("bind"); return 1;
    }
    /* Wake up every 200 ms so SIGINT can be noticed even with no traffic */
    struct timeval tv = { .tv_sec = 0, .tv_usec = 200000 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    printf("[receiver] listening on UDP port %d\n", port);

    uint8_t buf[RECV_BUF];
    struct sockaddr_in sender_addr;
    socklen_t sender_len = sizeof(sender_addr);

    /* Print bytes/sec every second */
    struct timespec last_report;
    clock_gettime(CLOCK_MONOTONIC, &last_report);
    uint64_t bytes_since_last = 0;

    while (g_running) {
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                             (struct sockaddr *)&sender_addr, &sender_len);
        if (n < (ssize_t)sizeof(pkt_hdr_t)) continue;

        pkt_hdr_t hdr;
        memcpy(&hdr, buf, sizeof(pkt_hdr_t));

        DBG("pkt seq=%u data_len=%u\n", hdr.seq, hdr.data_len);

        /* Send ACK back to sender */
        ack_hdr_t ack;
        ack.seq = hdr.seq;
        sendto(sock, &ack, sizeof(ack), 0,
               (struct sockaddr *)&sender_addr, sender_len);

        g_total_bytes  += hdr.data_len;
        bytes_since_last += hdr.data_len;

        /* Report throughput every second */
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (double)(now.tv_sec  - last_report.tv_sec) +
                         (double)(now.tv_nsec - last_report.tv_nsec) / 1e9;
        if (elapsed >= 1.0) {
            printf("[receiver] %.2f MB/s  total=%.2f MB\n",
                   (double)bytes_since_last / elapsed / 1e6,
                   (double)g_total_bytes / 1e6);
            bytes_since_last = 0;
            last_report = now;
        }
    }

    printf("[receiver] exiting. total bytes received: %llu\n",
           (unsigned long long)g_total_bytes);
    close(sock);
    return 0;
}
