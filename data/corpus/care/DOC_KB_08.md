---
doc_id: DOC_KB_08
title: Wi-Fi disconnects on specific networks
type: kb
---

## §1 Symptom

The device drops from a particular Wi-Fi network while other devices stay connected, or
refuses to reconnect after sleep. This is almost always a router-side band-steering or
security-mode incompatibility rather than a device fault.

## §2 Checks

Forget the network and reconnect. Disable Wi-Fi power saving. Ask whether the router
uses WPA3 only; devices older than One UI 5 may require a mixed WPA2/WPA3 mode. Test on
a mobile hotspot to establish whether the fault follows the device or the network.

## §3 Escalation

Only escalate where the fault reproduces on a hotspot and on a second independent
network. Quote diagnostic code NET-030. Connectivity complaints that reproduce on only
one network are not accepted as hardware faults.
