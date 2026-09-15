import React, { useEffect, useState } from 'react'
import { useRipple } from './useRipple.js'

/**
 * STARTING POINT for Member C. The shipping UI is ../dashboard.html; this is
 * the migration target. Port panels one at a time and keep both working —
 * dashboard.html is what the container falls back to.
 *
 * Still to build (see ../README.md for the full checklist):
 *   - StabilityChart: the signature graphic. Curve + θ line + decision markers.
 *   - VersionDiff: preserved / superseded / added / citation-drift pills.
 *     The drift counter must read 0 on every refinement — that is gate G5,
 *     visible at a glance.
 *   - EvidencePool: lifecycle states, colour-coded.
 *   - CitationPopover: /api/corpus?cite=... so a judge can click any citation
 *     and read the exact passage it rests on.
 */
export default function App() {
  const { connected, health, transcript, events, claims, turn, replay, reset } = useRipple()
  const [scenarios, setScenarios] = useState([])
  const [selected, setSelected] = useState('')
  const [speed, setSpeed] = useState(1)

  useEffect(() => {
    fetch('/api/scenarios').then(r => r.json()).then(s => {
      setScenarios(s)
      if (s.length) setSelected(s[0].scenario_id)
    })
  }, [])

  const ttft = turn?.ttft_rel_utterance_end

  return (
    <div className="min-h-screen">
      <header className="flex flex-wrap items-center gap-4 border-b border-line bg-panel px-4 py-2.5">
        <h1 className="text-[15px] font-semibold">
          Ripple <span className="text-accent">· Streaming Live RAG</span>
        </h1>
        <select value={selected} onChange={e => setSelected(e.target.value)}
          className="rounded border border-line bg-panel2 px-2 py-1 text-[13px]">
          {scenarios.map(s => (
            <option key={s.scenario_id} value={s.scenario_id}>{s.scenario_id}</option>
          ))}
        </select>
        <select value={speed} onChange={e => setSpeed(+e.target.value)}
          className="rounded border border-line bg-panel2 px-2 py-1 text-[13px]">
          <option value={1}>1× realtime</option>
          <option value={2}>2×</option>
          <option value={99}>instant</option>
        </select>
        <button onClick={() => replay(selected, speed)} disabled={!connected}
          className="rounded border border-line bg-panel2 px-3 py-1 text-[13px] hover:border-accent disabled:opacity-40">
          Replay
        </button>
        <button onClick={reset}
          className="rounded border border-line bg-panel2 px-3 py-1 text-[13px] hover:border-accent">
          Reset
        </button>
        <span className="font-mono text-[11px] text-ink3">
          {health
            ? `provider=${health.provider} · ${health.index.documents} docs / ${health.index.chunks} chunks · θ=${health.theta}`
            : 'connecting…'}
        </span>
      </header>

      <main className="grid gap-px bg-line lg:grid-cols-[1fr_1.25fr_1fr]">
        <Panel title="Live transcript">
          {transcript.map((m, i) => (
            <span key={i} className={m.speaker === 'agent' ? 'text-warn' : 'text-ink2'}>
              {m.text}{' '}
            </span>
          ))}
        </Panel>

        <Panel title="Controller & retrieval timeline">
          {/* TODO(C): StabilityChart goes here */}
          {events.filter(e => e.type !== 'chunk_received').slice(-60).map((e, i) => (
            <div key={i} className="grid grid-cols-[52px_1fr] gap-2 border-b border-line py-1 text-[12.5px]">
              <div className="text-right font-mono text-[11px] text-ink3">
                {e.t_rel_s.toFixed(2)}s
              </div>
              <div className="min-w-0">
                <span className="mr-1.5 font-mono text-[10px] text-accent">{e.type}</span>
                <span className="text-ink2">
                  {e.controller?.reason || e.sub_query || (e.retrieved || []).join(' ')}
                </span>
              </div>
            </div>
          ))}
        </Panel>

        <Panel title="Answer">
          <div className="mb-3 grid grid-cols-2 gap-1.5">
            <Kpi label="TTFT vs utt. end" good={ttft < 0}
              value={ttft == null ? '—' : `${ttft >= 0 ? '+' : ''}${ttft.toFixed(2)}s`} />
            <Kpi label="answer version" value={turn ? `v${turn.version}` : '—'} />
            <Kpi label="retrievals" value={turn?.retrievals_this_turn ?? 0} />
            <Kpi label="citation drift"
              good={turn?.change_summary?.citations_changed_on_preserved === 0}
              value={turn?.change_summary?.citations_changed_on_preserved ?? '—'} />
          </div>
          {turn?.uncertainty && (
            <div className="mb-3 rounded border border-warn bg-[#2a2114] p-2.5 text-[12.5px] text-warn">
              <b>Uncertainty</b><br />{turn.uncertainty}
            </div>
          )}
          {claims.map(c => (
            <div key={c.id} className="mb-2 border-l-2 border-good bg-panel px-2.5 py-1.5">
              <div>{c.text}</div>
              <div className="mt-1">
                {c.citations.map(x => (
                  <span key={x} className="mr-1.5 inline-block rounded bg-[#11282e] px-1 font-mono text-[10.5px] text-accent">
                    {x}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </Panel>
      </main>
    </div>
  )
}

function Panel({ title, children }) {
  return (
    <section className="max-h-[calc(100vh-47px)] overflow-y-auto bg-bg p-3">
      <h2 className="mb-2.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-ink3">
        {title}
      </h2>
      {children}
    </section>
  )
}

function Kpi({ label, value, good }) {
  return (
    <div className="rounded border border-line bg-panel px-2.5 py-2">
      <div className={`font-mono text-[19px] font-semibold ${good ? 'text-good' : ''}`}>{value}</div>
      <div className="mt-0.5 text-[9.5px] uppercase tracking-[0.11em] text-ink3">{label}</div>
    </div>
  )
}
