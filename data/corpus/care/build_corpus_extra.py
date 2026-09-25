#!/usr/bin/env python3
"""Care Knowledge Pack, part two.

Imported and merged by build_corpus.py. Split across two files only so each
stays readable; the shape is identical.

SYNTHETIC. Not Samsung material. Every figure invented.

Why the corpus needed to grow: at 21 documents and 60 sections, retrieval
metrics are close to meaningless — a top-10 result set covers a sixth of the
entire corpus, so recall@k is high for uninteresting reasons and a distractor
has almost nowhere to hide. This part takes the pack past 60 documents and 250
sections, which is enough for a reranker to matter and for a wrong answer to be
genuinely available.

Everything added here preserves the engineered properties:

  * a SECOND contradiction pair, DOC_POL_21 superseded by DOC_POL_23, so
    supersession is not a single lucky case
  * more region-scoped documents, so the cross-border constraint has more than
    one claim to invalidate
  * many more lexical distractors: "battery" now appears across eight
    documents, "S24" across six, "flicker" across four
  * near-duplicate SLA and policy text across regional variants
  * the coverage holes stay OPEN. Nothing here covers screen-protector
    reimbursement, trade-in valuation, insurance claims or financing. Do not
    add them; the abstention demo depends on their absence.
"""

EXTRA = {}


def xdoc(doc_id, title, dtype, sections, effective_from=None, region=None,
         supersedes=None):
    EXTRA[doc_id] = dict(
        doc_id=doc_id, title=title, type=dtype, sections=sections,
        effective_from=effective_from, region=region,
        supersedes=supersedes or [],
    )


# ===========================================================================
# Symptom knowledge base
# ===========================================================================

xdoc("DOC_KB_04", "Touchscreen registers phantom input", "kb", [
    ("1", "Symptom", """
     The display responds to touches the user did not make, or ignores touches
     in a specific region. Often described as the keyboard typing by itself.
     Most reports follow a screen protector being fitted or a drop."""),
    ("2", "First-line checks", """
     Remove any screen protector and clean the display with a dry microfibre
     cloth. Enable Show Taps under Developer options and observe for one
     minute. Taps appearing with no finger contact indicate a digitiser fault.
     Taps that register in the wrong position indicate a calibration issue
     that a restart usually clears."""),
    ("3", "Escalation", """
     Escalate for display assembly replacement when phantom input reproduces
     with no protector fitted and after a restart. Quote diagnostic code
     TCH-051. Where the device has visible impact damage the repair falls
     under DOC_WAR_05 §1 and is chargeable unless an extended plan
     applies."""),
    ]
)

xdoc("DOC_KB_05", "Charging port not recognising the cable", "kb", [
    ("1", "Symptom", """
     The device does not charge, charges only at certain cable angles, or
     reports Moisture detected when the port is dry. Lint compaction in the
     port is the most common cause and is not a manufacturing defect."""),
    ("2", "Port inspection", """
     Inspect the port with a light. Compacted lint appears as a grey pad at
     the base of the connector. It can be removed with a wooden toothpick;
     never use metal. After cleaning, test with two known-good cables before
     concluding anything about the port."""),
    ("3", "Moisture detection", """
     A persistent Moisture detected warning on a dry port indicates corrosion
     on the connector pins and is treated as liquid ingress under DOC_WAR_05
     §1. The liquid damage indicator must be checked and the finding recorded
     before a quotation is issued."""),
    ("4", "Escalation", """
     Escalate for charging assembly replacement when the port is clean, two
     cables fail, and no corrosion is present. Quote diagnostic code CHG-019.
     Charging assemblies are variant specific; see DOC_PRT_02 §3 for the
     implications on imported devices."""),
    ]
)

xdoc("DOC_KB_06", "Camera fails to focus or shows a blurred region", "kb", [
    ("1", "Symptom", """
     One camera produces persistently soft images, or a fixed blurred patch
     appears in the same position across every photograph. The fixed patch
     usually indicates debris behind the lens cover rather than a sensor
     fault."""),
    ("2", "Checks", """
     Test each lens individually in Pro mode. Clean the exterior lens glass.
     Where a fixed patch survives cleaning, the debris is internal and
     requires a camera module replacement. Where all lenses are soft, check
     whether a case with a magnetic ring is interfering with optical image
     stabilisation."""),
    ("3", "Escalation", """
     Quote diagnostic code CAM-077 for internal debris and CAM-082 for
     stabilisation failure. Camera module replacement is covered under the
     standard warranty where no impact damage is present."""),
    ]
)

xdoc("DOC_KB_08", "Wi-Fi disconnects on specific networks", "kb", [
    ("1", "Symptom", """
     The device drops from a particular Wi-Fi network while other devices stay
     connected, or refuses to reconnect after sleep. This is almost always a
     router-side band-steering or security-mode incompatibility rather than a
     device fault."""),
    ("2", "Checks", """
     Forget the network and reconnect. Disable Wi-Fi power saving. Ask whether
     the router uses WPA3 only; devices older than One UI 5 may require a
     mixed WPA2/WPA3 mode. Test on a mobile hotspot to establish whether the
     fault follows the device or the network."""),
    ("3", "Escalation", """
     Only escalate where the fault reproduces on a hotspot and on a second
     independent network. Quote diagnostic code NET-030. Connectivity
     complaints that reproduce on only one network are not accepted as
     hardware faults."""),
    ]
)

xdoc("DOC_KB_09", "Call audio is faint or distorted", "kb", [
    ("1", "Symptom", """
     The earpiece is quiet, muffled, or crackles during calls while the
     loudspeaker works normally. Dust in the earpiece mesh accounts for most
     reports."""),
    ("2", "Checks", """
     Inspect and gently brush the earpiece mesh. Test with the loudspeaker and
     with a wired headset to isolate which audio path is affected. Run the
     Loudspeaker and Earpiece tests in Samsung Members and record both
     results."""),
    ("3", "Escalation", """
     Escalate when the earpiece test fails with a clean mesh. Quote diagnostic
     code AUD-064. Where both earpiece and loudspeaker fail, the fault is more
     likely on the mainboard and the quotation falls under the mainboard
     labour tier in DOC_POL_09 §1."""),
    ]
)

xdoc("DOC_KB_10", "Fingerprint recognition failing after a screen repair",
     "kb", [
    ("1", "Symptom", """
     The ultrasonic fingerprint sensor fails to enrol or rejects a previously
     working fingerprint, most often immediately after a display replacement
     or after a third-party protector is fitted."""),
    ("2", "Remediation", """
     Delete all enrolled fingerprints and re-enrol with the protector in
     place. The sensor calibrates to whatever sits above it, so enrolling
     without a protector and then fitting one reliably breaks recognition.
     See DOC_PRT_05 §2 for the accessory interaction."""),
    ("3", "Escalation", """
     Where re-enrolment fails on a clean display, the sensor calibration data
     may not have been written during the repair. Return the device to the
     centre that performed the repair; recalibration is included in the
     original repair and is not separately chargeable."""),
    ]
)

xdoc("DOC_KB_12", "Storage full warnings with little user data", "kb", [
    ("1", "Symptom", """
     The device reports insufficient storage while the user's own files
     account for a small fraction of capacity. Cached application data and
     orphaned update packages are the usual cause."""),
    ("2", "Remediation", """
     Use Device care to clear cached data. Remove downloaded update packages
     under Software update. Where a single application holds several gigabytes
     of cache, clearing that application's storage from Settings resolves it
     without data loss to the user's documents."""),
    ]
)

xdoc("DOC_KB_13", "Device restarts unexpectedly", "kb", [
    ("1", "Symptom", """
     Spontaneous restarts with no user action, either at random or reliably
     during a specific activity such as camera use or gaming."""),
    ("2", "Diagnosis", """
     Restarts during a single application point to that application. Restarts
     across unrelated activities, or during charging, point to power delivery
     and are more serious. Boot into Safe Mode and use the device normally for
     a day; no restarts in Safe Mode means a third-party application is
     responsible."""),
    ("3", "Escalation", """
     Escalate where restarts reproduce in Safe Mode. Quote diagnostic code
     SYS-011. Devices restarting during charging must also be checked against
     DOC_KB_11 §3 for thermal symptoms before being returned to the
     customer."""),
    ]
)

xdoc("DOC_KB_14", "Display brightness limited or dimming unexpectedly", "kb", [
    ("1", "Symptom", """
     Maximum brightness is lower than expected, or the display dims during
     sustained use. Thermal management dims the panel deliberately above a
     threshold; this is designed behaviour and not a fault."""),
    ("2", "Distinguishing behaviour from fault", """
     Expected: dimming during video recording, navigation in direct sunlight,
     or fast charging, recovering within minutes of the load stopping. Faulty:
     brightness capped from boot, or dimming that never recovers. Only the
     second pattern is escalated."""),
    ("3", "Escalation", """
     Escalate a permanently capped panel under diagnostic code DSP-121. Do not
     confuse this with the intermittent flicker in DOC_KB_01, which has a
     different cause and a different remedy."""),
    ]
)

xdoc("DOC_KB_15", "Software update fails partway through", "kb", [
    ("1", "Symptom", """
     An update downloads but fails to install, or the device reboots into
     recovery with an installation error. Insufficient free storage is the
     most common cause."""),
    ("2", "Remediation", """
     Ensure at least eight gigabytes free, charge above fifty percent, and
     retry over Wi-Fi. Where the failure repeats, wipe the cache partition
     from recovery. A factory reset is a last resort and requires the customer
     to have completed the backup obligations in DOC_POL_14 §1."""),
    ("3", "Escalation", """
     Devices that fail to complete an update after a cache wipe may have
     damaged storage. Escalate under diagnostic code STO-004. Do not attempt
     repeated flashing at the counter; each failed attempt risks the user
     partition."""),
    ]
)

xdoc("DOC_KB_16", "Battery swelling and physical deformation", "kb", [
    ("1", "Symptom", """
     The back cover lifts, the display is pushed out of the frame, or the
     device rocks when placed on a flat surface. This indicates battery
     swelling and is a safety matter."""),
    ("2", "Immediate handling", """
     Do not charge the device. Do not attempt to compress or puncture the
     battery. Place the device in an open area away from flammable material
     and arrange immediate service. A swollen battery is replaced regardless
     of warranty status; the chargeable element, if any, is the labour."""),
    ("3", "Reporting", """
     Every swelling case is logged to the regional service desk within
     twenty-four hours with the device serial number, purchase date and
     charging history. This reporting obligation does not depend on whether
     the customer proceeds with the repair."""),
    ]
)

xdoc("DOC_KB_17", "Bluetooth audio stuttering with wireless earbuds", "kb", [
    ("1", "Symptom", """
     Audio drops or stutters with wireless earbuds while the same earbuds work
     with another source. Codec negotiation and interference account for
     nearly all cases."""),
    ("2", "Checks", """
     Disable Dual Audio. Set the Bluetooth audio codec to SBC in Developer
     options as a test. Move away from 2.4 gigahertz interference sources such
     as microwave ovens and congested Wi-Fi. Re-pair the earbuds after
     clearing the previous pairing from both devices."""),
    ("3", "Escalation", """
     Escalate only where stuttering persists with a second known-good
     Bluetooth accessory. Quote diagnostic code BT-025."""),
    ]
)


# ===========================================================================
# Warranty
# ===========================================================================

xdoc("DOC_WAR_06", "Dead on arrival and early-life failures", "warranty", [
    ("1", "Definition", """
     A device that fails within seven days of purchase with a confirmed
     hardware fault is treated as dead on arrival. Early-life failure covers
     the period from eight to thirty days."""),
    ("2", "Handling", """
     Dead on arrival devices are returned to the point of sale for exchange,
     not repaired at a service centre. The service centre issues a technical
     confirmation report that the retailer requires before exchanging. Between
     eight and thirty days the device is repaired under the standard warranty
     unless the retailer's own return policy is more favourable."""),
    ("3", "Exclusions", """
     Physical or liquid damage found on inspection removes the device from the
     dead-on-arrival process entirely, whatever the age. The finding is
     recorded with photographs before the device is returned."""),
    ],
    region="domestic",
)

xdoc("DOC_WAR_07", "Coverage for refurbished and exchange units", "warranty", [
    ("1", "Coverage period", """
     A refurbished device supplied as a warranty exchange carries the remainder
     of the original device's warranty or ninety days, whichever is longer. A
     refurbished device purchased outright carries a full twelve-month
     warranty from the date of that purchase."""),
    ("2", "Identification", """
     Exchange units are identified by a serial number prefix recorded against
     the original service request. Where a customer disputes that a device is
     refurbished, the service request history is authoritative."""),
    ]
)

xdoc("DOC_WAR_08", "Battery replacement outside the degradation threshold",
     "warranty", [
    ("1", "When a battery is chargeable", """
     A battery measuring at or above eighty percent of design capacity is
     performing within specification. Replacement is available on request but
     is chargeable at the parts price in DOC_PRT_03 §2 plus display-tier
     labour, because the display must be removed to reach it."""),
    ("2", "Customer expectation", """
     Explain that capacity loss is gradual and expected, and that a device
     reaching eighty-two percent after eighteen months is behaving normally.
     Quoting the measured figure from the Samsung Members diagnostic avoids
     most disputes."""),
    ]
)

xdoc("DOC_WAR_09", "Transfer of warranty on resale", "warranty", [
    ("1", "Transferability", """
     The standard limited warranty follows the device, not the purchaser, and
     runs from the date of first retail sale. A second-hand buyer is covered
     for whatever remains, provided the original tax invoice is available."""),
    ("2", "Extended plans", """
     Extended protection plans are not transferable and terminate on resale.
     This differs from the standard warranty and is a frequent source of
     confusion; state it explicitly before quoting any repair."""),
    ]
)

xdoc("DOC_WAR_10", "Commercial and fleet device coverage", "warranty", [
    ("1", "Scope", """
     Devices purchased on a business account and deployed as a fleet are
     covered by the standard limited warranty unless a separate enterprise
     agreement applies. Where one does, its terms override this document and
     the account manager must be consulted before quoting."""),
    ("2", "Bulk handling", """
     Fleet repairs are booked as a batch against the business account rather
     than individually. Turnaround commitments in DOC_POL_11 apply per device,
     not per batch."""),
    ],
    region="domestic",
)


# ===========================================================================
# Policy, including the second contradiction pair
# ===========================================================================

xdoc("DOC_POL_21", "Accessory and in-box item replacement charges", "policy", [
    ("1", "Charging basis", """
     In-box accessories replaced outside their coverage period are charged at
     the accessory list price plus a flat handling fee of two hundred rupees
     per item. Cables and adapters are stocked at all centres; earphones are
     ordered on demand."""),
    ("2", "Bundling", """
     Where an accessory is replaced during a device repair, the handling fee
     is waived. This waiver applies once per service request regardless of the
     number of accessories involved."""),
    ],
    effective_from="2025-03-01",
)

xdoc("DOC_POL_23", "Amendment to accessory replacement charges", "policy", [
    ("1", "Revised charging basis", """
     With effect from 1 May 2026 this section replaces DOC_POL_21 §1.
     Accessories replaced outside their coverage period are charged at the
     accessory list price with no handling fee. The two hundred rupee handling
     fee is withdrawn entirely. Ordering lead times for non-stocked accessories
     are unchanged at five working days."""),
    ("2", "Scope of the amendment", """
     The bundling waiver in DOC_POL_21 §2 is now redundant and is withdrawn
     with it. No other provision of DOC_POL_21 is affected."""),
    ],
    effective_from="2026-05-01",
    supersedes=["DOC_POL_21#1", "DOC_POL_21#2"],
)

xdoc("DOC_POL_16", "Use of refurbished parts in repairs", "policy", [
    ("1", "When refurbished parts are used", """
     Refurbished parts may be used for out-of-warranty repairs where a new
     part is unavailable or where the customer elects the lower price. The
     customer must be told before work begins and the quotation must state it
     explicitly."""),
    ("2", "Warranty on the repair", """
     A repair using a refurbished part carries a ninety-day warranty on that
     part, the same as a repair using a new part. The device's underlying
     warranty status is unchanged by the choice."""),
    ]
)

xdoc("DOC_POL_17", "Repair warranty on completed work", "policy", [
    ("1", "Coverage", """
     Every completed repair carries a ninety-day warranty covering the parts
     replaced and the workmanship. A recurrence of the same fault within that
     period is repaired at no charge, including labour."""),
    ("2", "What is not covered", """
     A different fault, or the same fault arising from new physical damage, is
     a new repair and is quoted normally. Where there is disagreement about
     whether a fault is a recurrence, the original technician's notes and the
     diagnostic codes on both visits decide it."""),
    ]
)

xdoc("DOC_POL_18", "Handling devices left uncollected", "policy", [
    ("1", "Notification", """
     Customers are contacted on completion and again at seven and twenty-one
     days. Contact attempts are logged against the service request with date
     and channel."""),
    ("2", "Storage and disposal", """
     Devices uncollected after ninety days from the first notification may be
     disposed of in accordance with local law, after a final written notice to
     the registered address. Storage charges are not levied."""),
    ]
)

xdoc("DOC_POL_19", "Priority handling for accessibility needs", "policy", [
    ("1", "Eligibility", """
     Customers who identify a disability that makes being without the device
     materially harder are moved to the front of the walk-in queue and
     prioritised for loaner allocation under DOC_POL_15 §1, irrespective of
     whether they hold an extended plan."""),
    ("2", "Evidence", """
     No documentation is required. The customer's statement is sufficient and
     must not be questioned at the counter."""),
    ]
)

xdoc("DOC_POL_20", "Quotation revision after work begins", "policy", [
    ("1", "When a revision is permitted", """
     Where a second fault is found after an approved repair has started, work
     stops and a revised quotation is issued. The customer may approve the
     revision, decline the additional work, or withdraw the device with only
     the originally approved work completed."""),
    ("2", "What may not be revised", """
     The labour tier quoted at approval is fixed. A revision may add parts but
     may not reprice labour for work already approved, even where the repair
     proves more difficult than expected."""),
    ]
)

xdoc("DOC_POL_22", "Service turnaround during declared peak periods", "policy",
     [
    ("1", "Applicability", """
     During periods declared as peak by the regional service operation,
     typically following a major product launch or a festival season, the
     walk-in commitment in DOC_POL_11 §1 extends from four working hours to
     eight."""),
    ("2", "Customer communication", """
     Peak-period timings must be stated at the point of booking, not on
     collection. The automatic escalation trigger in DOC_POL_11 §3 is
     unchanged at seven working days."""),
    ],
    region="domestic",
)

xdoc("DOC_POL_24", "Handling customer-supplied parts", "policy", [
    ("1", "Position", """
     Service centres do not fit parts supplied by the customer. A device
     presented with a customer-supplied part for fitting is declined, and the
     reason is recorded on the service request."""),
    ("2", "Consequence for existing coverage", """
     Declining to fit a customer part does not affect the device's warranty.
     A device on which such a part has already been fitted elsewhere falls
     under DOC_WAR_05 §3."""),
    ]
)

xdoc("DOC_POL_25", "Recording and disclosing liquid damage findings", "policy",
     [
    ("1", "Procedure", """
     The liquid damage indicator is checked on every device presented with an
     electrical fault and the result photographed before disassembly. The
     photograph is attached to the service request."""),
    ("2", "Disclosure", """
     Where the indicator is triggered, the customer is shown the photograph
     and told that DOC_WAR_05 §1 applies before any quotation is discussed.
     A customer who disputes the finding in writing has the case referred to
     the regional desk under DOC_SVC_04 §2."""),
    ]
)


# ===========================================================================
# Parts
# ===========================================================================

xdoc("DOC_PRT_06", "Charging assembly parts and prices", "parts", [
    ("1", "Part numbers", """
     Galaxy S24 charging assembly GH96-S24-CA2, S24 Plus GH96-S24P-CA2, S24
     Ultra GH96-S24U-CA2. The assembly includes the connector, the flex cable
     and the microphone; the connector is not separately serviceable."""),
    ("2", "List prices and variant coding", """
     Indicative list prices excluding labour are two thousand four hundred
     rupees for the S24 and two thousand nine hundred for the Plus and Ultra.
     Unlike display assemblies, charging assemblies are coded by original
     market and must be ordered against the device's market code, which is why
     imported devices attract the lead times in DOC_POL_12 §2."""),
    ]
)

xdoc("DOC_PRT_07", "Camera module parts and prices", "parts", [
    ("1", "Part numbers", """
     S24 Ultra main camera module GH96-S24U-CM1, ultrawide GH96-S24U-CM2,
     periscope telephoto GH96-S24U-CM3. Modules are supplied calibrated and
     must not be swapped between devices."""),
    ("2", "List prices", """
     Indicative list prices excluding labour are eleven thousand two hundred
     rupees for the main module, four thousand eight hundred for the
     ultrawide, and fourteen thousand six hundred for the periscope
     telephoto."""),
    ]
)

xdoc("DOC_PRT_08", "Back glass and frame parts", "parts", [
    ("1", "Part numbers and scope", """
     Back glass is supplied as a panel with adhesive, part GH82-S24-BG1 and
     variants. The frame is not separately serviceable; frame damage requires
     a chassis replacement, which is usually uneconomic against the device's
     residual value."""),
    ("2", "List prices", """
     Indicative list prices excluding labour are four thousand six hundred
     rupees for the S24 back glass and six thousand one hundred for the Ultra.
     Back glass damage is impact damage and falls under DOC_WAR_05 §1."""),
    ]
)

xdoc("DOC_PRT_09", "Speaker and receiver parts", "parts", [
    ("1", "Part numbers", """
     Loudspeaker module GH96-S24-SPK1, earpiece receiver GH96-S24-RCV1. Both
     are common across the S24 family and are stocked at all centres."""),
    ("2", "List prices", """
     Indicative list prices excluding labour are one thousand one hundred
     rupees for the loudspeaker module and nine hundred for the earpiece
     receiver. Both are accessory-tier labour under DOC_POL_09 §1."""),
    ]
)

xdoc("DOC_PRT_10", "Chargers, cables and power adapters", "parts", [
    ("1", "Compatibility", """
     Certified 25-watt and 45-watt adapters are interchangeable across the S24
     family; the device negotiates the rate it supports. A 45-watt adapter
     does not damage a device that supports only 25 watts. Cables must be
     rated 5A for charging above 25 watts."""),
    ("2", "List prices", """
     Indicative list prices are one thousand six hundred rupees for the
     45-watt adapter, one thousand two hundred for the 25-watt, and four
     hundred fifty for a 5A cable. Charging accessories replaced outside
     coverage are charged under DOC_POL_23 §1."""),
    ]
)


# ===========================================================================
# Service process
# ===========================================================================

xdoc("DOC_SVC_05", "Changing or cancelling a service appointment", "service", [
    ("1", "Process", """
     Appointments can be changed or cancelled through the same channel used to
     book, up to two hours before the slot. Inside two hours the slot is
     released to walk-ins and the customer must rebook."""),
    ("2", "Repeated no-shows", """
     Three no-shows within ninety days restricts the account to walk-in
     service for the following ninety days. The customer is told when the
     restriction is applied, not afterwards."""),
    ]
)

xdoc("DOC_SVC_06", "Courier return for completed repairs", "service", [
    ("1", "Availability", """
     Where a customer cannot collect, a completed repair may be couriered to
     the address on the service request at the customer's cost, quoted before
     dispatch. The device is insured in transit by the courier's standard
     cover."""),
    ("2", "Exclusions", """
     Courier return is not offered to an address different from the one on the
     service request, nor for devices serviced under DOC_WAR_03, because the
     import documentation must be returned in person."""),
    ],
    region="domestic",
)

xdoc("DOC_SVC_07", "What a technical confirmation report contains", "service", [
    ("1", "Contents", """
     A technical confirmation report states the device model, serial number,
     the fault reproduced, the diagnostic codes recorded, and whether the
     fault is a manufacturing defect. It does not state a market value or a
     recommendation about replacement."""),
    ("2", "When it is issued", """
     Issued on request for dead-on-arrival exchanges under DOC_WAR_06 §2, for
     insurance-adjacent purposes where a third party requires it, and where a
     customer is escalating under DOC_SVC_04. The centre does not correspond
     with third parties directly."""),
    ]
)

xdoc("DOC_SVC_08", "Walk-in hours and queue management", "service", [
    ("1", "Hours", """
     Authorised centres open 10:00 to 19:00 Monday to Saturday. The last
     walk-in token is issued ninety minutes before closing so that a same-day
     repair can be completed."""),
    ("2", "Queue priority", """
     The order is: booked appointments, accessibility priority under
     DOC_POL_19 §1, then walk-ins by token. Fleet batch drop-offs under
     DOC_WAR_10 §2 are handled outside the queue by prior arrangement."""),
    ]
)

xdoc("DOC_SVC_09", "Explaining a declined warranty claim", "service", [
    ("1", "What must be said", """
     A declined claim is explained with the specific clause relied on, the
     diagnostic evidence, and the chargeable alternative with a price. A
     decline communicated without a clause reference is not complete and will
     be overturned on escalation."""),
    ("2", "Customer options", """
     The customer may accept the chargeable repair, decline and pay the
     diagnostic fee under DOC_POL_07 §2, or escalate under DOC_SVC_04. All
     three must be stated; offering only the first is a process failure."""),
    ]
)


# ===========================================================================
# Cross-border, to give the region constraint more to act on
# ===========================================================================

xdoc("DOC_WAR_11", "Language and documentation for imported devices",
     "warranty", [
    ("1", "Accepted documents", """
     Import evidence in a language other than English is accepted with a
     translation the customer arranges. The service centre does not translate
     and does not pay for translation. Untranslated documents are held under
     DOC_WAR_03 §3 rather than rejected."""),
    ("2", "Device language and region lock", """
     A device set to a language the technician cannot read is tested after the
     customer sets it to English. Some regional variants restrict features
     that are available locally; this is not a fault and is not repaired."""),
    ],
    region="cross-border",
)

xdoc("DOC_POL_26", "Parts sourcing for non-local market variants", "policy", [
    ("1", "Ordering", """
     Variant-specific parts for devices sold in another market are ordered
     through the regional parts desk rather than local stock. Where the
     variant was never certified locally the order is refused and the case is
     handled under DOC_WAR_03 §5."""),
    ("2", "Cost to the customer", """
     Freight on a variant-specific part ordered from another market is passed
     to the customer and must appear as a separate line on the quotation. It
     is not included in the labour tiers in DOC_POL_09 §1."""),
    ],
    region="cross-border",
)


# ===========================================================================
# STILL NOT PRESENT, DELIBERATELY. Do not add.
#   * screen protector reimbursement or replacement cost
#   * trade-in device valuation
#   * insurance claim procedure (DOC_SVC_07 §2 mentions third parties but
#     describes no procedure -- this is a near-miss on purpose)
#   * financing, EMI or instalment plans
# ===========================================================================
