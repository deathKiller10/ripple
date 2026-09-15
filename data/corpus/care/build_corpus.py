#!/usr/bin/env python3
"""Authors the Care Knowledge Pack.

This is a SYNTHETIC corpus written for this project. It is not Samsung
material and makes no claim to describe real Samsung policy. Every figure in
it is invented. It exists to be a *test instrument*: each document is here
because it exercises a specific mechanism, and the mapping is recorded in
data/corpus/care/CORPUS_DESIGN.md.

Engineered properties (see the architecture brief, section 6):
  * multi-intent spread      one situation spans KB + warranty + policy + parts
  * late-constraint override DOC_WAR_03 overrides DOC_WAR_01 once purchase
                             region is known
  * genuine contradiction    DOC_POL_09 supersedes DOC_POL_07 with dates
  * lexical distractors      "flicker" in a monitor doc, "S24" in an accessory doc
  * near-duplicates          DOC_POL_12 restates DOC_POL_11 with different SLAs
  * deliberate coverage hole nothing anywhere covers screen-protector
                             reimbursement or trade-in valuation
"""

import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))

DOCS = {}


def doc(doc_id, title, dtype, sections, effective_from=None, region=None,
        supersedes=None):
    DOCS[doc_id] = dict(
        doc_id=doc_id, title=title, type=dtype, sections=sections,
        effective_from=effective_from, region=region, supersedes=supersedes or [],
    )


# ---------------------------------------------------------------------------
# Symptom knowledge base
# ---------------------------------------------------------------------------

doc("DOC_KB_01", "Display flicker after a firmware update (Galaxy S series)",
    "kb", [
    ("1", "Symptom", """
     Intermittent horizontal flicker or brightness pulsing on the main display,
     reported most often within seven days of a firmware update. The flicker is
     usually visible at low brightness and on dark user interface themes. It
     affects Galaxy S23 and Galaxy S24 family devices running One UI 6.1 and
     later. Devices showing a persistent green or pink vertical band are a
     different fault; see DOC_KB_07."""),
    ("2", "First-line checks", """
     Ask the customer to disable Adaptive Brightness and Extra Dim, then observe
     for two minutes at 20 percent brightness. If the flicker stops, the cause
     is the adaptive brightness curve and a settings reset resolves it. If the
     flicker persists, boot the device into Safe Mode. Flicker that disappears
     in Safe Mode indicates a third-party application overlay, not a panel
     fault."""),
    ("3", "Software remediation", """
     Where the flicker survives Safe Mode, check for a pending display firmware
     patch under Settings, Software update. Build G991BXXU9EWL2 and later
     contain the corrected panel timing table. If the device is already on a
     corrected build, run the Display diagnostic in Samsung Members and record
     the result code before escalating."""),
    ("4", "Escalation to hardware", """
     Escalate to a service centre when the flicker reproduces in Safe Mode on a
     corrected firmware build, or when the Display diagnostic returns code
     DSP-114 or DSP-118. These codes indicate a display driver integrated
     circuit fault and require a display assembly replacement. Do not attempt
     further software remediation once either code is returned."""),
    ]
)

doc("DOC_KB_02", "Screen flicker on monitors and external displays",
    "kb", [
    ("1", "Symptom", """
     Visible flicker, tearing or brightness pulsing on a Samsung monitor or on
     an external display connected to a Galaxy device via DeX. This article
     covers standalone monitors and external panels only. For flicker on a
     phone or tablet built-in display see DOC_KB_01."""),
    ("2", "Cable and refresh rate", """
     Flicker on an external panel is most commonly a refresh-rate mismatch or a
     failing cable. Set the refresh rate to 60 hertz and test with a certified
     cable no longer than two metres. Adaptive Sync and variable refresh rate
     should be disabled during testing because they produce a flicker that is
     easily mistaken for a panel fault."""),
    ("3", "Panel self-test", """
     Disconnect all inputs and power the monitor on. If the self-test pattern
     is stable, the panel is serviceable and the fault lies upstream. If the
     self-test pattern itself flickers, book the monitor for service under the
     monitor warranty terms, which differ from mobile device terms."""),
    ]
)

doc("DOC_KB_03", "Rapid battery drain following a software update",
    "kb", [
    ("1", "Symptom", """
     Battery discharges substantially faster than before an update, commonly
     described as the device reaching twenty percent by early afternoon under
     normal use. Expected behaviour after a major update is elevated drain for
     forty-eight to seventy-two hours while the system re-indexes media and
     re-optimises applications."""),
    ("2", "Diagnosis", """
     Open Settings, Battery, and review usage since last full charge. Drain
     concentrated in Android System or Media Storage during the first three
     days after an update is expected and self-resolving. Drain concentrated in
     a single third-party application indicates a background wake-lock and is
     resolved by restricting that application's background activity."""),
    ("3", "Battery health assessment", """
     Run the Battery diagnostic in Samsung Members. A reported battery capacity
     below eighty percent of design capacity is considered degraded. Degraded
     batteries are covered for twelve months from the date of purchase under
     the standard limited warranty, which is a shorter period than the general
     hardware coverage described in DOC_WAR_01 §2."""),
    ("4", "Remediation", """
     Where drain persists beyond seventy-two hours with no dominant application
     and battery health is above eighty percent, perform a cache partition wipe
     and re-test for one full charge cycle. If drain still persists, escalate
     for battery replacement quoting diagnostic code BAT-207."""),
    ]
)

doc("DOC_KB_07", "Green or pink vertical band on AMOLED panels",
    "kb", [
    ("1", "Symptom", """
     A persistent vertical band, usually green but occasionally pink, appearing
     down one side of the display. The band is present from boot and does not
     change with brightness or content. This is a panel-level defect and is
     distinct from the intermittent flicker described in DOC_KB_01."""),
    ("2", "Handling", """
     No software remediation exists. Book a display assembly replacement
     directly. Where the device is within the standard warranty period and
     shows no physical or liquid damage, the replacement is covered. Devices
     with an extended protection plan follow the terms in DOC_WAR_04."""),
    ]
)

doc("DOC_KB_11", "Device overheating while charging",
    "kb", [
    ("1", "Symptom", """
     The device becomes uncomfortably warm during charging, or charging stops
     with an on-screen temperature warning. Warmth during fast charging is
     normal; a temperature warning that halts charging is not."""),
    ("2", "Checks", """
     Confirm the customer is using a certified charger and cable rated for the
     device's charging profile. Remove any case during testing, particularly
     magnetic or metal-backed cases. Charging on a bed or sofa restricts
     airflow and reproduces the warning on healthy devices."""),
    ("3", "Escalation", """
     Escalate when the warning reproduces with a certified charger, no case,
     and on a hard surface. Quote diagnostic code THM-042. Devices showing
     visible swelling must not be charged and require immediate service
     regardless of warranty status."""),
    ]
)


# ---------------------------------------------------------------------------
# Warranty
# ---------------------------------------------------------------------------

doc("DOC_WAR_01", "Standard limited warranty, mobile devices", "warranty",
    [
    ("1", "Scope", """
     This limited warranty applies to mobile phones and tablets purchased new
     from an authorised channel and used in the country of purchase. It covers
     defects in materials and workmanship under normal use. It does not create
     any obligation in respect of devices purchased outside the country of
     service; those are governed by DOC_WAR_03."""),
    ("2", "Coverage period", """
     Hardware is covered for twelve months from the date of original purchase.
     In-box accessories are covered for six months. Batteries and chargers are
     covered for twelve months but are assessed against the degradation
     threshold in DOC_KB_03 §3 rather than as a simple defect."""),
    ("3", "What is covered", """
     Covered repairs include display assembly failure, battery capacity below
     the degradation threshold, charging port failure, camera module failure
     and audio component failure, where each arises without external cause.
     Covered repairs are performed at no charge for parts or labour at an
     authorised service centre."""),
    ("4", "Proof of purchase", """
     A valid tax invoice showing the device serial number or IMEI and the date
     of purchase is required. Where no invoice is available, coverage is
     calculated from the date of manufacture plus three months, which is
     usually less favourable to the customer. Retailer order confirmations
     without a serial number are not accepted."""),
    ],
    region="domestic",   # explicitly "used in the country of purchase" (§1)
)

doc("DOC_WAR_03", "Cross-border and international purchase coverage",
    "warranty", [
    ("1", "When this document applies", """
     This document governs any device presented for service in a country other
     than the country where it was originally purchased. It applies whenever a
     customer says the device was bought abroad, bought overseas, bought while
     travelling, brought back from another country, purchased duty free at an
     airport, or bought in a specific foreign city or market such as Dubai,
     Singapore, London or New York. It also applies to devices bought online
     from a seller outside the country of service. Where it applies, it
     overrides the corresponding provisions of DOC_WAR_01 §1 and §2. Agents
     must establish the country of purchase before quoting any warranty
     outcome."""),
    ("2", "Coverage of imported devices", """
     Devices purchased in another country are serviced under this policy at the
     discretion of the local service operation. Coverage is limited to twelve
     months from original purchase, as with domestic devices, but parts
     availability is not guaranteed and the customer must accept an extended
     turnaround. Model variants not sold locally may be declined entirely."""),
    ("3", "Additional documentation", """
     In addition to the proof of purchase required by DOC_WAR_01 §4, the
     customer must supply evidence of lawful import: a customs declaration, a
     travel-purchase receipt, or a duty-paid certificate. Claims submitted
     without import evidence are held rather than rejected, and the customer is
     given twenty-one days to supply it."""),
    ("4", "Foreign currency and reimbursement", """
     Where a repair is chargeable and the original purchase was made in a
     foreign currency, the invoice is raised in local currency at the reference
     rate on the date of service. Any reimbursement to the customer requires
     verification of the original foreign-currency receipt by the regional
     finance desk, which adds five to seven working days."""),
    ("5", "Excluded regions", """
     Devices originally sold in a market where the model was never certified
     for local use cannot be serviced and must be declined with an explanation.
     The current list of non-serviceable model and market combinations is
     maintained by the regional service operation and reviewed quarterly."""),
    ],
    region="cross-border",
)

doc("DOC_WAR_04", "Extended protection plan terms", "warranty", [
    ("1", "Plan scope", """
     The extended protection plan is an optional paid plan purchased within
     thirty days of the device. It runs for twenty-four months from device
     purchase and sits on top of the standard limited warranty rather than
     replacing it. Where the two documents differ, the more favourable term to
     the customer applies."""),
    ("2", "Accidental damage", """
     The plan covers accidental damage including cracked displays and liquid
     ingress, which the standard warranty excludes under DOC_WAR_05 §1. Each
     accidental damage claim carries a fixed service fee which varies by device
     tier and is payable before the repair begins. The plan permits two
     accidental damage claims in any twelve-month period."""),
    ("3", "Interaction with imported devices", """
     The extended plan is sold and honoured on a country basis. A plan
     purchased in one country does not transfer with the device to another. A
     customer presenting an imported device with a foreign extended plan is
     handled under DOC_WAR_03 and the foreign plan is not honoured
     locally."""),
    ]
)

doc("DOC_WAR_05", "Warranty exclusions", "warranty", [
    ("1", "Excluded causes", """
     The standard limited warranty excludes accidental damage, cracked or
     shattered displays arising from impact, liquid ingress, damage from
     non-certified chargers, cosmetic wear, and any fault arising after repair
     by an unauthorised party. Devices with a triggered liquid damage indicator
     are treated as liquid ingress unless the customer disputes the finding in
     writing."""),
    ("2", "Software and data", """
     The warranty covers hardware only. Software faults, operating system
     behaviour, application compatibility and data loss are outside its scope.
     Where a covered hardware repair requires a device reset, the data handling
     obligations in DOC_POL_14 apply."""),
    ("3", "Unauthorised modification", """
     Devices with an unlocked bootloader, modified system software or a
     replaced component sourced outside the authorised channel are excluded
     from warranty coverage for any fault plausibly related to that
     modification. Unrelated faults may still be covered at the discretion of
     the service centre."""),
    ]
)


# ---------------------------------------------------------------------------
# Policy, including the deliberate contradiction pair
# ---------------------------------------------------------------------------

doc("DOC_POL_07", "Out-of-warranty repair pricing policy", "policy", [
    ("1", "Pricing basis", """
     Out-of-warranty repairs are quoted as the list price of the parts required
     plus a flat labour charge of nine hundred rupees per repair. The quotation
     is valid for fourteen days. Customers must approve the quotation in
     writing before any work begins, and an unapproved device is returned
     unrepaired at no charge."""),
    ("2", "Diagnostic fee", """
     A diagnostic fee of four hundred rupees applies where the customer
     declines the quotation. The fee is waived where the repair proceeds, and
     waived entirely for devices presented within thirty days of a previous
     repair for the same fault."""),
    ("3", "Payment", """
     Payment is collected on collection of the device. Part payment or
     instalment arrangements are not offered at service centre level and must
     be arranged with the retailer."""),
    ],
    effective_from="2025-01-01",
)

doc("DOC_POL_09", "Amendment to out-of-warranty repair pricing", "policy", [
    ("1", "Revised pricing basis", """
     With effect from 1 April 2026 this section replaces DOC_POL_07 §1.
     Out-of-warranty repairs are quoted as the list price of the parts required
     plus a tiered labour charge: six hundred rupees for accessory and port
     repairs, one thousand two hundred rupees for display assembly
     replacement, and one thousand eight hundred rupees for mainboard work.
     The flat nine hundred rupee charge no longer applies."""),
    ("2", "Quotation validity", """
     Quotation validity is extended from fourteen days to thirty days. All
     other provisions of DOC_POL_07, including the diagnostic fee in §2 and the
     payment terms in §3, remain in force unamended."""),
    ],
    effective_from="2026-04-01",
    supersedes=["DOC_POL_07#1"],
)

doc("DOC_POL_11", "Service centre turnaround commitments", "policy", [
    ("1", "Standard turnaround", """
     Walk-in repairs with parts in stock are completed within four working
     hours. Repairs requiring a part order are completed within three working
     days of the part arriving. The service centre must contact the customer
     within twenty-four hours of any slippage against these commitments."""),
    ("2", "Display assembly repairs", """
     Display assembly replacement for current-generation flagship models is
     stocked at all authorised centres and is a same-day repair. Older
     generations and regional variants may require a part order with a three to
     five working day lead time."""),
    ("3", "Escalation triggers", """
     Any repair open for more than seven working days is automatically
     escalated to the regional service manager. Customers may request a status
     review at any point through the process in DOC_SVC_04."""),
    ],
    region="domestic",   # DOC_POL_12 explicitly overrides these for imports
)

doc("DOC_POL_12", "Service turnaround for imported and grey-market devices",
    "policy", [
    ("1", "Applicability", """
     This document sets the turnaround commitments that apply instead of
     DOC_POL_11 §1 and §2 where the device was purchased outside the country of
     service, as established under DOC_WAR_03 §1."""),
    ("2", "Turnaround", """
     Imported devices are not covered by the four working hour walk-in
     commitment. Parts for regional variants are ordered on demand with a lead
     time of ten to fifteen working days, and in some cases longer where the
     variant is not certified locally. The customer must be told this lead time
     before the device is accepted."""),
    ("3", "Escalation", """
     The seven working day automatic escalation in DOC_POL_11 §3 does not apply
     to imported devices. The equivalent trigger is twenty-one working
     days."""),
    ],
    region="cross-border",
)

doc("DOC_POL_14", "Data backup and privacy before service", "policy", [
    ("1", "Customer obligation", """
     Customers are responsible for backing up their data before handing over a
     device. Service centres are not able to recover data from a device that
     fails during repair and offer no data recovery service."""),
    ("2", "Account removal", """
     The customer must remove their account and disable device protection
     before the device is accepted. A device that arrives with device
     protection active cannot be tested after repair and will be returned
     unrepaired."""),
    ]
)

doc("DOC_POL_15", "Loaner device availability", "policy", [
    ("1", "Eligibility", """
     A loaner device is offered where a covered repair is expected to exceed
     three working days and the customer holds an extended protection plan.
     Loaner stock is limited and allocated on a first-come basis at each
     centre."""),
    ("2", "Exclusions", """
     Loaner devices are not offered for out-of-warranty repairs, for imported
     devices serviced under DOC_WAR_03, or where the customer has an
     outstanding balance on a previous repair."""),
    ],
    region="domestic",   # DOC_POL_15 §2 excludes imported devices
)


# ---------------------------------------------------------------------------
# Parts and pricing
# ---------------------------------------------------------------------------

doc("DOC_PRT_02", "Display assembly parts and list prices, S24 family",
    "parts", [
    ("1", "Part numbers", """
     Galaxy S24 display assembly, part GH82-S24-DA1, includes the panel, the
     digitiser and the pre-fitted frame. Galaxy S24 Plus uses GH82-S24P-DA1 and
     Galaxy S24 Ultra uses GH82-S24U-DA1. Assemblies are supplied with adhesive
     and are not separable into component parts."""),
    ("2", "List prices", """
     Indicative list prices excluding labour are eighteen thousand nine hundred
     rupees for the S24 assembly, twenty-two thousand four hundred rupees for
     the S24 Plus and twenty-nine thousand six hundred rupees for the S24
     Ultra. Prices are reviewed quarterly and the quotation issued at the
     service centre is authoritative."""),
    ("3", "Regional variants", """
     Display assemblies are common across regional variants for the S24 family,
     so an imported S24 can be repaired with locally stocked parts. This is not
     true of all models; mainboards and charging assemblies are variant
     specific and must be ordered against the device's original market
     code."""),
    ]
)

doc("DOC_PRT_03", "Battery parts and list prices, current generation",
    "parts", [
    ("1", "Part numbers", """
     Galaxy S24 battery pack EB-BS921ABY, Galaxy S24 Plus EB-BS926ABY, Galaxy
     S24 Ultra EB-BS928ABY. Battery replacement requires display removal on all
     three models and is therefore quoted with display-tier labour under
     DOC_POL_09 §1."""),
    ("2", "List prices", """
     Indicative list prices excluding labour are three thousand four hundred
     rupees for the S24 pack and three thousand nine hundred rupees for the
     Plus and Ultra packs. Batteries replaced under warranty against the
     degradation threshold carry no parts or labour charge."""),
    ]
)

doc("DOC_PRT_05", "Accessory compatibility, Galaxy S24 series cases and covers",
    "parts", [
    ("1", "Case compatibility", """
     Galaxy S24 cases are not cross-compatible with S24 Plus or S24 Ultra
     owing to differing camera module cut-outs and body dimensions. Cases sold
     for the S23 series do not fit the S24 series. Magnetic accessory rings
     fitted to a case may interfere with wireless charging alignment."""),
    ("2", "Screen protection accessories", """
     Factory-applied screen film is a shipping protection layer and is not a
     screen protector. Third-party tempered glass fitted over the display may
     affect ultrasonic fingerprint recognition; recognition failures with a
     third-party protector fitted are resolved by re-registering the
     fingerprint with the protector in place."""),
    ("3", "In-box accessory coverage", """
     In-box accessories carry the six month coverage stated in DOC_WAR_01 §2.
     Separately purchased accessories carry their own twelve month coverage
     from the accessory purchase date, evidenced by a separate invoice."""),
    ]
)


# ---------------------------------------------------------------------------
# Service process
# ---------------------------------------------------------------------------

doc("DOC_SVC_01", "Booking a service appointment", "service", [
    ("1", "Channels", """
     Appointments are booked through Samsung Members, the support website or by
     calling the contact centre. Walk-in service is available at all authorised
     centres but is served after booked appointments and may involve a
     wait."""),
    ("2", "What the customer must bring", """
     The device, the proof of purchase required by DOC_WAR_01 §4, and a
     government photograph identity document. For an imported device the
     additional import evidence in DOC_WAR_03 §3 is also required at the time
     of booking, not at collection."""),
    ]
)

doc("DOC_SVC_02", "Doorstep pickup service", "service", [
    ("1", "Availability", """
     Doorstep pickup is available in selected metropolitan pin codes for
     in-warranty repairs on flagship devices. The service adds two working days
     to the turnaround commitments in DOC_POL_11 to allow for collection and
     return transit."""),
    ("2", "Exclusions", """
     Doorstep pickup is not offered for out-of-warranty repairs, for devices
     with suspected liquid damage, or for imported devices serviced under
     DOC_WAR_03. Customers in these categories must attend a service
     centre."""),
    ],
    region="domestic",   # DOC_SVC_02 §2 excludes imported devices
)

doc("DOC_SVC_04", "Escalation and complaint handling", "service", [
    ("1", "First escalation", """
     A customer dissatisfied with a service outcome may request escalation to
     the service centre manager, who must respond within two working days. The
     request is logged against the original service request number."""),
    ("2", "Regional escalation", """
     Where the first escalation does not resolve the complaint, the case is
     referred to the regional service desk with a five working day response
     commitment. Cases involving a declined warranty claim on an imported
     device are referred directly to the regional desk, bypassing the first
     stage."""),
    ]
)


# ---------------------------------------------------------------------------
# NOT PRESENT, DELIBERATELY:
#   * screen protector reimbursement or replacement cost
#   * trade-in device valuation
#   * insurance claim procedure
# These are the coverage holes that force an explicit uncertainty indicator.
# ---------------------------------------------------------------------------


def write():
    written = []
    for doc_id, d in DOCS.items():
        lines = ["---", f"doc_id: {d['doc_id']}", f"title: {d['title']}",
                 f"type: {d['type']}"]
        if d["effective_from"]:
            lines.append(f"effective_from: {d['effective_from']}")
        if d["region"]:
            lines.append(f"region: {d['region']}")
        if d["supersedes"]:
            lines.append("supersedes: " + ", ".join(d["supersedes"]))
        lines.append("---")
        lines.append("")
        for sec, sec_title, body in d["sections"]:
            lines.append(f"## §{sec} {sec_title}")
            lines.append("")
            lines.append(textwrap.fill(" ".join(body.split()), width=88))
            lines.append("")
        path = os.path.join(HERE, f"{doc_id}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        written.append(doc_id)
    return written


if __name__ == "__main__":
    w = write()
    nsec = sum(len(d["sections"]) for d in DOCS.values())
    print(f"wrote {len(w)} documents, {nsec} sections")
