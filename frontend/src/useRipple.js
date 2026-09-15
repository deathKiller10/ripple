import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * One hook, one WebSocket, all engine state.
 *
 * Deliberately dumb: it accumulates telemetry events and exposes them. It does
 * not derive metrics. Every number the UI displays must come from an event the
 * engine emitted, so that the screen and traces/<session>.jsonl cannot disagree.
 */
export function useRipple() {
  const ws = useRef(null)
  const [connected, setConnected] = useState(false)
  const [health, setHealth] = useState(null)
  const [transcript, setTranscript] = useState([])
  const [events, setEvents] = useState([])
  const [stability, setStability] = useState([])
  const [claims, setClaims] = useState([])
  const [pool, setPool] = useState([])
  const [turn, setTurn] = useState(null)

  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(setHealth).catch(() => {})
  }, [])

  const connect = useCallback(() => {
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://'
    const sock = new WebSocket(proto + location.host + '/ws')
    sock.onopen = () => setConnected(true)
    sock.onclose = () => { setConnected(false); setTimeout(connect, 1200) }
    sock.onmessage = e => {
      const m = JSON.parse(e.data)
      if (m.type === 'transcript') { setTranscript(t => [...t, m]); return }
      if (m.type === 'turn') { setTurn(m); return }
      if (m.type === 'state') { setClaims(m.claims); setPool(m.pool); return }
      if (m.t_rel_s === undefined) return
      setEvents(ev => [...ev, m])
      if (m.type === 'controller_decision' && m.controller?.stability !== undefined) {
        setStability(s => [...s, {
          t: m.t_rel_s, s: m.controller.stability, d: m.controller.decision,
        }])
      }
    }
    ws.current = sock
  }, [])

  useEffect(() => { connect() }, [connect])

  const clear = () => {
    setTranscript([]); setEvents([]); setStability([])
    setClaims([]); setPool([]); setTurn(null)
  }

  const replay = (scenarioId, speed = 1) => {
    clear()
    ws.current?.send(JSON.stringify({ type: 'replay', scenario_id: scenarioId, speed }))
  }
  const reset = () => { clear(); ws.current?.send(JSON.stringify({ type: 'reset' })) }

  return { connected, health, transcript, events, stability, claims, pool, turn, replay, reset }
}
