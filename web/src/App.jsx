import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

// The FastAPI backend (server.py) — started separately with:
//   ./.venv/bin/python3 -m uvicorn server:app --port 8000
const API_BASE = 'http://localhost:8000'

function RetrievedChunks({ chunks }) {
  const [open, setOpen] = useState(false)
  const [expandedId, setExpandedId] = useState(null)

  return (
    <div className="trace">
      <button className="trace-toggle" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Retrieved {chunks.length} chunk{chunks.length !== 1 ? 's' : ''}
      </button>
      {open && (
        <ul className="chunk-list">
          {chunks.map((chunk) => (
            <li
              key={chunk.chunk_id}
              className={`chunk-item ${chunk.relevant === false ? 'chunk-dropped' : ''}`}
            >
              <button
                className="chunk-summary"
                onClick={() =>
                  setExpandedId(expandedId === chunk.chunk_id ? null : chunk.chunk_id)
                }
              >
                <span className="chunk-source">{chunk.source_file}</span>
                <span className="chunk-distance">distance={chunk.distance.toFixed(3)}</span>
                <span className={`badge badge-${chunk.doc_type}`}>{chunk.doc_type}</span>
                {chunk.employee_id && <span className="chunk-emp">{chunk.employee_id}</span>}
                {chunk.relevant === false && (
                  <span className="badge badge-dropped" title="Dropped by the relevance guardrail before reaching the LLM">
                    dropped
                  </span>
                )}
              </button>
              {expandedId === chunk.chunk_id && (
                <pre className="chunk-text">{chunk.text}</pre>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [userId, setUserId] = useState('')
  const [provider, setProvider] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/meta`)
      .then((r) => r.json())
      .then((data) => {
        setMeta(data)
        setUserId(data.default_user_id)
        setProvider(data.default_provider)
      })
      .catch(() => setError('Could not reach the backend at ' + API_BASE))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSubmit(e) {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    setInput('')
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', text: query, userId, provider }])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, user_id: userId, provider }),
      })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data = await res.json()
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          answer: data.answer,
          chunks: data.retrieved_chunks,
          guardrailTriggered: data.guardrail_triggered,
          // Server-resolved role, not the client's own belief about it --
          // this is what the access-control decision actually used.
          resolvedRole: data.role,
        },
      ])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'error', text: err.message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>HR RAG Assistant</h1>
        <div className="controls">
          <label>
            Logged in as
            <select value={userId} onChange={(e) => setUserId(e.target.value)}>
              {(meta?.users ?? []).map((u) => (
                <option key={u.id} value={u.id}>
                  {u.id} ({u.role})
                </option>
              ))}
            </select>
          </label>
          <label>
            Provider
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              {(meta?.providers ?? []).map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <main className="conversation">
        {messages.length === 0 && (
          <p className="empty-hint">
            Ask something like "How many casual leaves do I get?" — try switching "Logged
            in as" to EMP99999 (employee) and asking about Priya's offer letter to see the
            access-control filter block it, then switch to HR001 (hr) and ask the same
            question to see the RBAC override let it through.
          </p>
        )}

        {messages.map((m, i) => {
          if (m.role === 'user') {
            return (
              <div key={i} className="message user-message">
                <div className="bubble">{m.text}</div>
                <div className="message-meta">
                  as {m.userId} · via {m.provider}
                </div>
              </div>
            )
          }
          if (m.role === 'error') {
            return (
              <div key={i} className="message error-message">
                <div className="bubble">⚠ {m.text}</div>
              </div>
            )
          }
          return (
            <div key={i} className="message assistant-message">
              <RetrievedChunks chunks={m.chunks} />
              {m.guardrailTriggered && (
                <div className="guardrail-note">
                  ⚑ Relevance guardrail triggered — the LLM was not called; this answer is a fixed message
                </div>
              )}
              <div className="bubble markdown-bubble">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.answer}</ReactMarkdown>
              </div>
              <div className="message-meta">role used: {m.resolvedRole}</div>
            </div>
          )
        })}

        {loading && (
          <div className="message assistant-message">
            <div className="bubble bubble-loading">thinking…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      {error && <div className="banner-error">{error}</div>}

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask an HR question…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
