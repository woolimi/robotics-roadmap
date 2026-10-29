import { useEffect, useRef, useState } from 'react'
import { CONFIG } from '../config'

const P = CONFIG.panel

function storedWidth(): number {
  try {
    const v = Number(localStorage.getItem(CONFIG.storage.panelWidthKey))
    return v > 0 ? v : P.defaultWidth
  } catch {
    return P.defaultWidth
  }
}

function storeWidth(width: number) {
  try {
    localStorage.setItem(CONFIG.storage.panelWidthKey, String(width))
  } catch {
    // storage unavailable: the panel opens at the default width next visit
  }
}

export interface PanelWidth {
  width: number
  min: number
  max: number
  clamp(width: number): number
  /** clamps to [min, max]; `save` also remembers it for the next visit */
  setWidth(width: number, save?: boolean): void
}

/** The panel width the viewer chose, kept inside the window. */
export function usePanelWidth(): PanelWidth {
  const [preferred, setPreferred] = useState(storedWidth)
  const [viewport, setViewport] = useState(() => window.innerWidth)

  useEffect(() => {
    const onResize = () => setViewport(window.innerWidth)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  // never squeeze the panel below its default just because the window is narrow
  const max = Math.max(P.defaultWidth, viewport - P.edgeGutter)
  const clamp = (w: number) => Math.round(Math.min(Math.max(w, P.minWidth), max))
  const setWidth = (w: number, save = false) => {
    setPreferred(clamp(w))
    if (save) storeWidth(clamp(w))
  }
  return { width: clamp(preferred), min: P.minWidth, max, clamp, setWidth }
}

interface Props {
  panel: PanelWidth
  onDragging(dragging: boolean): void
}

/** Vertical splitter on the panel's left edge: drag, arrow keys, Home/End, double-click to reset. */
export function PanelResizer({ panel, onDragging }: Props) {
  const { width, min, max, clamp, setWidth } = panel
  // `last` is the width from the latest pointermove; React only hears about it when the pointer is released
  const drag = useRef<{ x: number; width: number; last: number; app: HTMLElement; frame: number } | null>(null)

  // the panel can close mid-drag (the window turning narrow) and no pointerup reaches us then
  useEffect(() => () => onDragging(false), [onDragging])

  const endDrag = () => {
    if (!drag.current) return
    const { last, frame } = drag.current
    cancelAnimationFrame(frame)
    drag.current = null
    onDragging(false)
    setWidth(last, true)
  }

  return (
    <div
      className="panel-resizer"
      role="separator"
      aria-orientation="vertical"
      aria-label="강의 창 너비"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={width}
      tabIndex={0}
      title="드래그해서 너비 조절 · 두 번 클릭하면 기본 너비"
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.currentTarget.setPointerCapture(e.pointerId)
        drag.current = { x: e.clientX, width, last: width, app: e.currentTarget.parentElement!, frame: 0 }
        onDragging(true)
      }}
      onPointerMove={(e) => {
        if (!drag.current) return
        // the panel is anchored to the right edge, so moving left widens it
        const d = drag.current
        d.last = clamp(d.width + d.x - e.clientX)
        // write the CSS variable straight to the DOM, once per frame: re-rendering the app on every pointermove
        // re-renders the whole graph and the lesson body
        cancelAnimationFrame(d.frame)
        d.frame = requestAnimationFrame(() => d.app.style.setProperty('--panel-width', `${d.last}px`))
      }}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={() => setWidth(P.defaultWidth, true)}
      onKeyDown={(e) => {
        const keys: Record<string, number> = {
          ArrowLeft: width + P.keyStep,
          ArrowRight: width - P.keyStep,
          Home: min,
          End: max,
        }
        const next = keys[e.key]
        if (next === undefined) return
        e.preventDefault()
        setWidth(next, true)
      }}
    />
  )
}
