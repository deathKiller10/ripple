---
doc_id: DOC_KB_01
title: Display flicker after a firmware update (Galaxy S series)
type: kb
---

## §1 Symptom

Intermittent horizontal flicker or brightness pulsing on the main display, reported most
often within seven days of a firmware update. The flicker is usually visible at low
brightness and on dark user interface themes. It affects Galaxy S23 and Galaxy S24
family devices running One UI 6.1 and later. Devices showing a persistent green or pink
vertical band are a different fault; see DOC_KB_07.

## §2 First-line checks

Ask the customer to disable Adaptive Brightness and Extra Dim, then observe for two
minutes at 20 percent brightness. If the flicker stops, the cause is the adaptive
brightness curve and a settings reset resolves it. If the flicker persists, boot the
device into Safe Mode. Flicker that disappears in Safe Mode indicates a third-party
application overlay, not a panel fault.

## §3 Software remediation

Where the flicker survives Safe Mode, check for a pending display firmware patch under
Settings, Software update. Build G991BXXU9EWL2 and later contain the corrected panel
timing table. If the device is already on a corrected build, run the Display diagnostic
in Samsung Members and record the result code before escalating.

## §4 Escalation to hardware

Escalate to a service centre when the flicker reproduces in Safe Mode on a corrected
firmware build, or when the Display diagnostic returns code DSP-114 or DSP-118. These
codes indicate a display driver integrated circuit fault and require a display assembly
replacement. Do not attempt further software remediation once either code is returned.
