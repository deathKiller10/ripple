---
doc_id: DOC_KB_03
title: Rapid battery drain following a software update
type: kb
---

## §1 Symptom

Battery discharges substantially faster than before an update, commonly described as the
device reaching twenty percent by early afternoon under normal use. Expected behaviour
after a major update is elevated drain for forty-eight to seventy-two hours while the
system re-indexes media and re-optimises applications.

## §2 Diagnosis

Open Settings, Battery, and review usage since last full charge. Drain concentrated in
Android System or Media Storage during the first three days after an update is expected
and self-resolving. Drain concentrated in a single third-party application indicates a
background wake-lock and is resolved by restricting that application's background
activity.

## §3 Battery health assessment

Run the Battery diagnostic in Samsung Members. A reported battery capacity below eighty
percent of design capacity is considered degraded. Degraded batteries are covered for
twelve months from the date of purchase under the standard limited warranty, which is a
shorter period than the general hardware coverage described in DOC_WAR_01 §2.

## §4 Remediation

Where drain persists beyond seventy-two hours with no dominant application and battery
health is above eighty percent, perform a cache partition wipe and re-test for one full
charge cycle. If drain still persists, escalate for battery replacement quoting
diagnostic code BAT-207.
