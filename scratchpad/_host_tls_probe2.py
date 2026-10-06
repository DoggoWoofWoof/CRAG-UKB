"""Host TLS probe 2: decode the issuer/subject of the leaf and every chain cert a few hosts present (metadata only)."""
import socket, ssl, os, tempfile

for h in ["huggingface.co", "example.com"]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection((h, 443), timeout=15) as s:
        with ctx.wrap_socket(s, server_hostname=h) as t:
            chain = []
            try:
                chain = [ssl.DER_cert_to_PEM_cert(c) for c in t.get_unverified_chain()] if hasattr(t, "get_unverified_chain") else []
            except Exception as e:
                print("chain err", e)
            if not chain:
                chain = [ssl.DER_cert_to_PEM_cert(t.getpeercert(binary_form=True))]
    for i, pem in enumerate(chain):
        f = os.path.join(tempfile.gettempdir(), "probe_%d.pem" % i)
        open(f, "w").write(pem)
        d = ssl._ssl._test_decode_cert(f)
        sub = " / ".join("%s=%s" % x[0] for x in d["subject"])
        iss = " / ".join("%s=%s" % x[0] for x in d["issuer"])
        print(h, "cert", i, "| subject:", sub, "| issuer:", iss, "| notBefore:", d.get("notBefore"))
