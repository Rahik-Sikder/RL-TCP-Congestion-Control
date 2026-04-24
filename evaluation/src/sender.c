#include <arpa/inet.h>
#include <getopt.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "cc/cc.h"
#include "cc/cubic.h"
#include "cc/new_reno.h"
#include "cc/ppo.h"
#include "transport/transport.h"

#ifdef DEBUG
#define DBG(...) fprintf(stderr, "[sender] " __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --cc ppo|newreno|cubic [--model DIR]\n"
        "          --dest-ip IP --port PORT\n"
        "          [--bytes N] [--out-dir DIR] [--scenario NAME]\n",
        prog);
    exit(1);
}

static void write_summary(const char *out_dir, const char *algo,
                           const char *scenario, const char *model_path,
                           transport_t *t, double elapsed_sec) {
    char path[512];
    snprintf(path, sizeof(path), "%s/summary.json", out_dir);
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen summary.json"); return; }

    double avg_rtt = t->rtt_count > 0
        ? t->rtt_sum_ms / (double)t->rtt_count : 0.0;
    double loss_rate = t->total_sent > 0
        ? (double)(t->total_sent - t->total_acked) / (double)t->total_sent
        : 0.0;
    /* throughput: acked_pkts * MSS * 8 bits / elapsed_sec / 1e6 = Mbps (approximate) */
    double throughput_mbps = elapsed_sec > 0
        ? ((double)t->total_acked * MSS * 8.0) / (elapsed_sec * 1e6) : 0.0;

    fprintf(f,
        "{\n"
        "  \"scenario\": \"%s\",\n"
        "  \"algorithm\": \"%s\",\n"
        "  \"model_path\": \"%s\",\n"
        "  \"avg_throughput_mbps\": %.4f,\n"
        "  \"avg_rtt_ms\": %.4f,\n"
        "  \"total_loss_rate\": %.6f,\n"
        "  \"total_packets_sent\": %llu,\n"
        "  \"total_packets_acked\": %llu,\n"
        "  \"total_timeouts\": %llu\n"
        "}\n",
        scenario, algo, model_path ? model_path : "",
        throughput_mbps, avg_rtt, loss_rate,
        (unsigned long long)t->total_sent,
        (unsigned long long)t->total_acked,
        (unsigned long long)t->total_timeouts);
    fclose(f);
    DBG("wrote %s\n", path);
}

int main(int argc, char **argv) {
    const char *cc_name     = NULL;
    const char *model_dir   = NULL;
    const char *dest_ip     = NULL;
    const char *out_dir     = "results";
    const char *scenario    = "unknown";
    int         port        = 5000;
    size_t      total_bytes = 10 * 1024 * 1024;  /* 10 MB default */

    static struct option long_opts[] = {
        {"cc",       required_argument, 0, 'c'},
        {"model",    required_argument, 0, 'm'},
        {"dest-ip",  required_argument, 0, 'd'},
        {"port",     required_argument, 0, 'p'},
        {"bytes",    required_argument, 0, 'b'},
        {"out-dir",  required_argument, 0, 'o'},
        {"scenario", required_argument, 0, 's'},
        {0, 0, 0, 0},
    };

    int ch;
    while ((ch = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (ch) {
        case 'c': cc_name     = optarg; break;
        case 'm': model_dir   = optarg; break;
        case 'd': dest_ip     = optarg; break;
        case 'p': port        = atoi(optarg); break;
        case 'b': total_bytes = (size_t)atoll(optarg); break;
        case 'o': out_dir     = optarg; break;
        case 's': scenario    = optarg; break;
        default:  usage(argv[0]);
        }
    }

    if (!cc_name || !dest_ip) usage(argv[0]);

    DBG("cc=%s dest=%s:%d bytes=%zu out=%s scenario=%s\n",
        cc_name, dest_ip, port, total_bytes, out_dir, scenario);

    /* UDP socket connected to receiver */
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t)port);
    addr.sin_addr.s_addr = inet_addr(dest_ip);
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect"); return 1;
    }
    DBG("connected to %s:%d\n", dest_ip, port);

    /* Initialize selected CC module */
    cc_ops_t *cc  = NULL;
    void     *ctx = NULL;

    if (strcmp(cc_name, "newreno") == 0) {
        cc = new_reno_create(&ctx);
    } else if (strcmp(cc_name, "cubic") == 0) {
        cc = cubic_create(&ctx);
    } else if (strcmp(cc_name, "ppo") == 0) {
        if (!model_dir) {
            fprintf(stderr, "--cc ppo requires --model DIR\n");
            return 1;
        }
        cc = ppo_create(&ctx, model_dir);
    } else {
        fprintf(stderr, "unknown --cc value: %s\n", cc_name);
        return 1;
    }

    if (!cc) {
        fprintf(stderr, "failed to initialize CC module '%s'\n", cc_name);
        return 1;
    }

    /* Open timeseries CSV */
    char ts_path[512];
    snprintf(ts_path, sizeof(ts_path), "%s/timeseries.csv", out_dir);
    FILE *ts_file = fopen(ts_path, "w");
    if (!ts_file) { perror("fopen timeseries.csv"); return 1; }

    /* Run transfer */
    transport_t *t = transport_create(sock, cc, ctx, ts_file);
    if (!t) { fprintf(stderr, "transport_create failed\n"); return 1; }

    DBG("starting %zu-byte transfer\n", total_bytes);
    uint64_t t_start = now_us();
    transport_run(t, total_bytes);
    uint64_t t_end   = now_us();
    double elapsed   = (double)(t_end - t_start) / 1e6;

    fclose(ts_file);

    printf("Done. algo=%s elapsed=%.2fs throughput=%.3fMbps "
           "sent=%llu acked=%llu timeouts=%llu\n",
           cc_name, elapsed,
           elapsed > 0 ? ((double)t->total_acked * MSS * 8.0) / (elapsed * 1e6) : 0.0,
           (unsigned long long)t->total_sent,
           (unsigned long long)t->total_acked,
           (unsigned long long)t->total_timeouts);

    write_summary(out_dir, cc_name, scenario, model_dir, t, elapsed);

    transport_destroy(t);
    cc->destroy(ctx);
    close(sock);
    return 0;
}
