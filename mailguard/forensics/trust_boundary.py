"""The trust boundary: which Received hops can be believed, and which cannot.

WHY THIS EXISTS

As an email travels, every mail server that handles it prepends a
Received header. Read bottom to top they look like a complete travel
history. They are not trustworthy. A Received header is just text, and any
server on the path can write anything it wants. An attacker sending from
their own machine can fabricate five plausible Received headers before the
mail ever leaves, describing a journey that never happened.

The only headers we can trust are the ones written by servers WE control,
because the attacker never touched those servers.

THE WALK

Start at the TOP of the chain (index 0, the hop closest to us, added by our
own mail server) and walk DOWNWARD. Each hop is trusted while its `by` host
matches a configured host we own or a relay we configured (a case
insensitive suffix match on a label boundary). The first hop that does not
match is the TRUST BOUNDARY. Everything at or below that index is attacker
writable and is reported as unverifiable, never believed.

One refinement closes the obvious attack on that rule. A trusted hop also
records who handed the message to our server: the peer's IP and the
reverse DNS name OUR server looked up. If that peer is not one of our own
machines (not an internal address, not a reverse DNS name in our trusted
list), the message crossed from the outside world at that hop, and the
walk stops there even if the next hop down claims `by mx.ourdomain`. Such
a claim sits below an external hand-off, so it can only have been written
outside our infrastructure: it is a forgery, and it is recorded as one.
The HELO name is never used for this, because the connecting client
chooses it.

THE IP WE CAN STAND BEHIND

The earliest address in the chain we can actually stand behind is the
peer IP recorded by our LAST TRUSTED hop, the socket address our own
server saw when the message crossed the boundary. boundary_ip() returns
that address and it becomes the origin candidate for attribution. The
`from` IP written inside the boundary hop itself was written by a server
we do not control, so it is a claim like every other hop below.

RETURN VALUES OF find_trust_boundary()

  None          empty chain, or every hop was written by our servers and the
                bottom hop received the message from an internal machine:
                the mail originated inside our infrastructure.
  0             not even the top hop is ours: nothing can be believed.
  1 .. n-1      the first attacker writable hop.
  n (= len)     every hop is ours but the bottom one received the message
                directly from an external peer: the sender added no hops of
                their own, and the boundary sits just below the chain.
"""
from __future__ import annotations

import ipaddress
from typing import Any, Optional

from mailguard.core.config import host_matches
from mailguard.core.models import ParsedEmail, ReceivedHop
from mailguard.parsing.header_parser import parse_received_line

# Networks reachable only from inside an organisation. Listed explicitly
# instead of using ipaddress.is_private, which also treats the
# documentation ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) as
# private, and those are what fixtures use for "some public address".
INTERNAL_NETWORKS: tuple[Any, ...] = tuple(
    ipaddress.ip_network(n)
    for n in (
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
        "127.0.0.0/8", "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10",
    )
)


def is_internal_ip(ip: Optional[str]) -> bool:
    """True for RFC 1918, CGNAT, loopback, link-local and unique-local addresses."""
    if not ip:
        return False
    try:
        address = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return False
    return any(address.version == net.version and address in net for net in INTERNAL_NETWORKS)


def _peer_is_ours(hop: ReceivedHop, trusted_hosts: list[str]) -> tuple[bool, str]:
    """Did this trusted hop receive the message from one of our own machines?"""
    parts = parse_received_line(hop.raw)
    ip = parts.get("ip") or hop.from_ip
    rdns = parts.get("rdns")
    if not ip and not rdns and not parts.get("helo") and not hop.from_host:
        return True, "no peer recorded (internal hand-off within one server)"
    if is_internal_ip(ip):
        return True, f"peer {ip} is an internal address"
    if rdns and host_matches(rdns, trusted_hosts):
        return True, f"peer reverse DNS {rdns} is one of ours"
    return False, f"peer {rdns or 'with no reverse DNS'} [{ip or 'no IP'}] is external"


def find_trust_boundary(chain: list[ReceivedHop], trusted_hosts: list[str]) -> Optional[int]:
    """Walk from index 0 downward and return the index of the first untrusted hop.

    Sets `hop.trusted` on every hop. See the module docstring for the
    meaning of each possible return value.
    """
    if not chain:
        return None
    for hop in chain:
        hop.trusted = False
    for index, hop in enumerate(chain):
        if not host_matches(hop.by_host, trusted_hosts):
            return index
        hop.trusted = True
        ours, _why = _peer_is_ours(hop, trusted_hosts)
        if not ours:
            return index + 1
    return None


def boundary_ip(chain: list[ReceivedHop], boundary_index: Optional[int]) -> Optional[str]:
    """The earliest IP we can stand behind: the peer our last trusted hop recorded.

    None when there is no boundary (internal mail, empty chain) or when the
    boundary is at hop 0 (no hop of ours recorded anything).
    """
    if boundary_index is None or boundary_index <= 0 or not chain:
        return None
    last_trusted = chain[min(boundary_index, len(chain)) - 1]
    return last_trusted.from_ip


def boundary_host(chain: list[ReceivedHop], boundary_index: Optional[int]) -> Optional[str]:
    """Reverse DNS name of the host that crossed the boundary, as our server looked it up.

    Only the reverse DNS name counts. The HELO name is chosen by the
    connecting client, so an attacker can introduce themselves as
    `mail-out.google.com` or as one of our own gateways; returning it here
    would let them pick their own attribution. None when our server
    recorded no reverse DNS ("unknown").
    """
    if boundary_index is None or boundary_index <= 0 or not chain:
        return None
    last_trusted = chain[min(boundary_index, len(chain)) - 1]
    return parse_received_line(last_trusted.raw).get("rdns")


def boundary_helo(chain: list[ReceivedHop], boundary_index: Optional[int]) -> Optional[str]:
    """The HELO name the boundary host CLAIMED. Unverified; reported, never matched on."""
    if boundary_index is None or boundary_index <= 0 or not chain:
        return None
    last_trusted = chain[min(boundary_index, len(chain)) - 1]
    return parse_received_line(last_trusted.raw).get("helo")


def annotate_chain(email: ParsedEmail, trusted_hosts: list[str]) -> ParsedEmail:
    """Set trust_boundary_index and every hop's `trusted` flag, then return the email.

    Also records a summary in email.meta["trust_boundary"]: the boundary
    IP and host, the IPs claimed below the boundary (recorded, never
    believed), and any attacker writable hop that names one of our own
    servers as `by`, which is a forgery.
    """
    chain = email.received_chain or []
    try:
        index = find_trust_boundary(chain, trusted_hosts)
    except Exception as exc:  # fail closed: nothing trusted
        for hop in chain:
            hop.trusted = False
        email.trust_boundary_index = 0 if chain else None
        email.meta["trust_boundary"] = {"error": f"{type(exc).__name__}: {exc}"}
        return email
    email.trust_boundary_index = index
    below = chain[index:] if index is not None else []
    email.meta["trust_boundary"] = {
        "index": index,
        "boundary_ip": boundary_ip(chain, index),
        "boundary_host": boundary_host(chain, index),
        "boundary_helo": boundary_helo(chain, index),
        "trusted_hops": sum(1 for hop in chain if hop.trusted),
        "total_hops": len(chain),
        "unverified_ips": [hop.from_ip for hop in below if hop.from_ip],
        "forged_internal_hops": [hop.index for hop in below if host_matches(hop.by_host, trusted_hosts)],
        "trusted_hosts": list(trusted_hosts),
    }
    return email


def describe_boundary(email: ParsedEmail) -> str:
    """One line human readable summary of the boundary, for the terminal and the report."""
    chain = email.received_chain or []
    index = email.trust_boundary_index
    if not chain:
        return "No Received headers: the route cannot be examined"
    if index is None:
        return "No trust boundary: every hop was written by our own servers (internal mail)"
    if index == 0:
        return (
            f"Trust boundary at hop 0: not even the top hop was written by our servers, "
            f"so all {len(chain)} hops are attacker writable"
        )
    ip = boundary_ip(chain, index) or "no IP recorded"
    forged = (email.meta.get("trust_boundary") or {}).get("forged_internal_hops") or []
    suffix = f"; hop {forged[0]} falsely claims to be our server" if forged else ""
    if index >= len(chain):
        return f"Trust boundary below hop {len(chain) - 1}: our server received it directly from {ip}{suffix}"
    return f"Trust boundary at hop {index}, everything below is attacker writable (origin candidate {ip}){suffix}"


if __name__ == "__main__":  # pragma: no cover - demonstration
    trusted = ["mx.example.org", "gw.example.org"]
    fake_chain = [
        ReceivedHop(0, "from gw.example.org (gw.example.org [10.0.0.5]) by mx.example.org; Fri, 18 Sep 2026 03:41:20 +0000",
                    "gw.example.org", "10.0.0.5", "mx.example.org"),
        ReceivedHop(1, "from mail.sender.example (mail.sender.example [203.0.113.19]) by gw.example.org; Fri, 18 Sep 2026 03:41:18 +0000",
                    "mail.sender.example", "203.0.113.19", "gw.example.org"),
        ReceivedHop(2, "from relay.fake.example (relay.fake.example [192.0.2.44]) by mail.sender.example; Fri, 18 Sep 2026 03:41:15 +0000",
                    "relay.fake.example", "192.0.2.44", "mail.sender.example"),
        ReceivedHop(3, "from hop.fake.example (hop.fake.example [192.0.2.45]) by relay.fake.example; Fri, 18 Sep 2026 03:41:12 +0000",
                    "hop.fake.example", "192.0.2.45", "relay.fake.example"),
        ReceivedHop(4, "from origin (unknown [10.9.9.9]) by hop.fake.example; Fri, 18 Sep 2026 03:41:10 +0000",
                    "origin", "10.9.9.9", "hop.fake.example"),
    ]
    index = find_trust_boundary(fake_chain, trusted)
    print(f"boundary index : {index}")
    print(f"boundary ip    : {boundary_ip(fake_chain, index)}")
    for hop in fake_chain:
        print(f"  [{hop.index}] {'trusted   ' if hop.trusted else 'UNVERIFIED'} by {hop.by_host}")
