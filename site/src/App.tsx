import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Legend, NextUp, TopBar, ZoomControls } from './components/Chrome'
import { Holygraph, type GraphControls, type Insets } from './components/Holygraph'
import { Intro } from './components/Intro'
import { PanelResizer, usePanelWidth } from './components/PanelResizer'
import { UpdateBanner } from './components/UpdateBanner'
import { CONFIG } from './config'
import { curriculum, layout } from './curriculum/load'
import { useProgress } from './curriculum/progress'

// the panel pulls in markdown, KaTeX and highlighting; the graph does not need them to show up
const loadPanel = () => import('./components/LessonPanel')
const LessonPanel = lazy(() => loadPanel().then((m) => ({ default: m.LessonPanel })))

const ROUTE = /^#\/p\/([a-z0-9-]+)$/

function readRoute(): string | null {
  const m = ROUTE.exec(window.location.hash)
  return m && curriculum.byId.has(m[1]) ? m[1] : null
}

function useMediaQuery(query: string): boolean {
  const [match, setMatch] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const on = () => setMatch(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [query])
  return match
}

function introSeen(): boolean {
  try {
    return localStorage.getItem(CONFIG.storage.introSeenKey) === '1'
  } catch {
    return false
  }
}

export function App() {
  // fetch the panel's code once the map is up, so the first lesson opens without waiting for it
  useEffect(() => {
    const idle = window.requestIdleCallback ?? ((f: () => void) => window.setTimeout(f, CONFIG.panel.prefetchDelay))
    idle(() => void loadPanel())
  }, [])
  const progressFile = useProgress()
  const [selectedId, setSelectedId] = useState<string | null>(readRoute)
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [hoverBranch, setHoverBranch] = useState<string | null>(null)
  const [showIntro, setShowIntro] = useState(() => !introSeen() && !readRoute())
  const controls = useRef<GraphControls>(null)
  const wide = useMediaQuery('(min-width: 960px)')
  const sidebarInset = wide ? 280 : 0
  const panel = usePanelWidth()
  const [resizing, setResizing] = useState(false)

  useEffect(() => {
    const onHash = () => setSelectedId(readRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const select = useCallback((id: string) => {
    window.location.hash = `#/p/${id}`
  }, [])
  const close = useCallback(() => {
    history.pushState(null, '', window.location.pathname + window.location.search + '#/')
    setSelectedId(null)
  }, [])

  const closeIntro = useCallback(() => {
    try {
      localStorage.setItem(CONFIG.storage.introSeenKey, '1')
    } catch {
      // storage unavailable: the intro shows again next visit
    }
    setShowIntro(false)
  }, [])

  const panelOpen = selectedId !== null
  const insets: Insets = {
    top: 64,
    right: panelOpen && wide ? panel.width : 0,
    bottom: wide ? 0 : panelOpen ? window.innerHeight * 0.55 : 72,
    left: sidebarInset,
  }

  const root = curriculum.lessons.find((p) => p.requires.length === 0 && p.branch === 'core') ?? curriculum.lessons[0]

  return (
    <div
      className={`app ${panelOpen ? 'has-panel' : ''} ${resizing ? 'is-resizing' : ''}`}
      style={{ ['--panel-width' as string]: `${panel.width}px` }}
    >
      <Holygraph
        ref={controls}
        curriculum={curriculum}
        layout={layout}
        progress={progressFile}
        selectedId={selectedId}
        hoverId={hoverId}
        hoverBranch={hoverBranch}
        insets={insets}
        onSelect={select}
        onBackgroundClick={() => selectedId && close()}
        onHover={setHoverId}
      />

      <TopBar
        curriculum={curriculum}
        progressFile={progressFile}
        onHelp={() => setShowIntro(true)}
      />

      <UpdateBanner />

      <div className="sidebar">
        <NextUp curriculum={curriculum} progressFile={progressFile} onSelect={select} onHover={setHoverId} />
        <Legend curriculum={curriculum} onHoverBranch={setHoverBranch} />
      </div>

      <ZoomControls controls={controls} />

      {selectedId && wide && <PanelResizer panel={panel} onDragging={setResizing} />}

      {selectedId && (
        <Suspense fallback={<aside className="panel" aria-busy="true" />}>
          <LessonPanel
            key={selectedId}
            curriculum={curriculum}
            progressFile={progressFile}
            lessonId={selectedId}
            onClose={close}
            onSelect={select}
          />
        </Suspense>
      )}

      {showIntro && (
        <Intro
          firstLessonId={root.id}
          lessonCount={curriculum.lessons.length}
          onClose={closeIntro}
          onStart={(id) => {
            closeIntro()
            select(id)
          }}
        />
      )}
    </div>
  )
}
