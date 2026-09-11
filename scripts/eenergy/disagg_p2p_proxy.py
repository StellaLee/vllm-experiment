# SPIKE (throwaway): wraps the vendored disagg_proxy_demo.py to fix a real gap for
# P2pNcclConnector specifically -- the generic reference proxy forwards requests unchanged,
# but P2pNcclConnector.parse_request_id() (vllm/distributed/kv_transfer/kv_connector/v1/p2p/
# p2p_nccl_connector.py) requires the request id to embed the counterpart instance's address:
#   prefill leg:  ___decode_addr_<host>:<port>
#   decode leg:   ___prefill_addr_<host>:<port>___
# (confirmed via source inspection of parse_request_id's regex, not guessed). vLLM's
# completions endpoint honors a caller-supplied "request_id" field
# (vllm/entrypoints/openai/completion/serving.py: request_id = f"cmpl-{...request.request_id}"),
# so this subclass just tags each leg's request_id with the required marker before forwarding.
#
# IMPORTANT (found on the 2nd attempt, via source inspection of P2pNcclConnector.save_kv_layer/
# inject_kv_into_layer): the embedded <port> is NOT the counterpart's OpenAI HTTP port -- the
# connector computes remote_address = parsed_ip + ":" + str(parsed_port + self._rank), where
# self._rank is this worker's own rank (0 for a non-TP single-GPU instance) and parsed_port must
# therefore be the counterpart's ZMQ transfer port (its --kv-transfer-config kv_port), not its
# --port. The 1st attempt embedded the HTTP ports (8100/8200) by mistake, which is why the
# request hung forever (the sender's ZMQ DEALER socket tried to speak ZMQ wire protocol to an
# HTTP server) and the decode instance's HTTP log flooded with "Invalid HTTP request received."
import sys
import uuid

sys.path.insert(0, "/root/pli/vllm-experiment/scripts/eenergy")
import disagg_proxy_demo as base  # noqa: E402

# Must match --kv-transfer-config's kv_port for each instance in run_disagg_energy_probe.sh.
PREFILL_KV_PORT = 21001
DECODE_KV_PORT = 21002

# P2pNcclEngine binds its ZMQ router socket to get_ip() (the box's real LAN-visible address)
# when no explicit hostname is configured -- NOT localhost/127.0.0.1, confirmed via `ss -tlnp`
# showing both engines listening on 10.0.0.16:21001/21002 while the OpenAI HTTP servers listen
# on 0.0.0.0:8100/8200. Embedding "localhost" here made the sender's ZMQ DEALER socket try to
# connect to a port nothing listens on; ZMQ retries silently instead of raising, which is why
# the 2nd attempt (right ports, wrong host) hung forever with zero log output on either side.
KV_TRANSFER_HOST = "10.0.0.16"


class P2PProxy(base.Proxy):
    async def create_completion(self, raw_request: base.Request):
        try:
            request = await raw_request.json()

            prefill_instance = self.schedule(self.prefill_cycler)
            decode_instance = self.schedule(self.decode_cycler)
            base_id = uuid.uuid4().hex

            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1
            kv_prepare_request["request_id"] = (
                f"{base_id}___decode_addr_{KV_TRANSFER_HOST}:{DECODE_KV_PORT}"
            )

            try:
                async for _ in self.forward_request(
                    f"http://{prefill_instance}/v1/completions", kv_prepare_request
                ):
                    continue
            except base.HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc

            request["request_id"] = (
                f"{base_id}___prefill_addr_{KV_TRANSFER_HOST}:{PREFILL_KV_PORT}___"
            )
            try:
                generator = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request
                )
            except base.HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            return base.StreamingResponse(generator)
        except Exception:
            import sys as _sys

            exc_info = _sys.exc_info()
            print("Error occurred in P2P disagg proxy server")
            print(exc_info)


class P2PProxyServer(base.ProxyServer):
    def __init__(self, args):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.proxy_instance = P2PProxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=base.RoundRobinSchedulingPolicy(),
        )


if __name__ == "__main__":
    args = base.parse_args()
    server = P2PProxyServer(args=args)
    server.run_server()
