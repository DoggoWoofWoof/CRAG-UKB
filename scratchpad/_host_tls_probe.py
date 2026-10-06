"""Host TLS probe: print the issuer chain each host presents (metadata only, no payload sent)."""
import socket, ssl, sys

HOSTS = ["huggingface.co", "cdn-lfs.hf.co", "cas-bridge.xethub.hf.co", "cas-server.xethub.hf.co",
         "s3.us-east-1.amazonaws.com", "github.com", "pypi.org", "example.com"]


def probe(h):
    out = {"host": h}
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((h, 443), timeout=15) as s:
            with ctx.wrap_socket(s, server_hostname=h) as t:
                out["verified"] = True
                out["issuer"] = dict(x[0] for x in t.getpeercert()["issuer"]).get("organizationName")
    except Exception as e:
        out["verified"] = False
        out["err"] = str(e)[:120]
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with socket.create_connection((h, 443), timeout=15) as s:
                with ctx.wrap_socket(s, server_hostname=h) as t:
                    der = t.getpeercert(binary_form=True)
                    out["leaf_der_bytes"] = len(der)
                    out["cipher"] = t.cipher()[0]
        except Exception as e2:
            out["err2"] = str(e2)[:120]
    return out


print("python", sys.version.split()[0], "openssl", ssl.OPENSSL_VERSION)
print("verify paths", ssl.get_default_verify_paths())
for h in HOSTS:
    print(probe(h))
