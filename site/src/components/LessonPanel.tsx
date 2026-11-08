import type { Root } from 'hast'
import 'katex/dist/katex.min.css'
import { BookOpen, ClipboardCheck, Clock, ExternalLink, ListChecks, X } from './icons'
import { memo, useEffect, useRef, useState } from 'react'
import { CONFIG } from '../config'
import { loadBody } from '../curriculum/load'
import { hasQuiz } from '../curriculum/load'
import { attemptsOf, nodeState, passedOn, statusOf, type ProgressFile } from '../curriculum/progress'
import type { Curriculum, NodeState, Status } from '../curriculum/types'
import { Markdown } from './Markdown'
import { QuizReturn, QuizSection, useQuizNav } from './Quiz'

interface Props {
  curriculum: Curriculum
  progressFile: ProgressFile
  lessonId: string
  onClose(): void
  onSelect(id: string): void
}

const STATE_TEXT: Record<NodeState, string> = {
  done: '완료',
  in_progress: '진행 중',
  available: '시작 가능',
  locked: '선수 강의 남음',
}

function formatHours(h: number): string {
  return `강의와 퀴즈 약 ${h}시간`
}

function LessonChip({ c, p, id, onSelect }: { c: Curriculum; p: ProgressFile; id: string; onSelect(id: string): void }) {
  const lesson = c.byId.get(id)!
  const branch = c.branches.find((b) => b.id === lesson.branch)!
  const state = nodeState(c, p, id)
  return (
    <li>
      <button
        type="button"
        className={`chip is-${state}`}
        style={{ ['--branch' as string]: branch.color }}
        onClick={() => onSelect(id)}
        title={STATE_TEXT[state]}
      >
        <span className="chip-dot" aria-hidden="true" />
        {lesson.title}
        <span className="sr-only">, {STATE_TEXT[state]}</span>
      </button>
    </li>
  )
}

function statusText(status: Status, passedDate: string | undefined, tries: number, quiz: boolean): string {
  if (status === 'done') return `퀴즈 통과 · ${passedDate}`
  if (status === 'in_progress') return `퀴즈 ${tries}회 응시 · 아직 통과 전`
  return quiz ? '강의를 끝까지 읽고 퀴즈로 마무리' : '퀴즈 준비 중'
}

/** Where the learner stands on this lesson, with a button that opens its quiz. */
function StatusCard({ progressFile, lessonId, onTake }: { progressFile: ProgressFile; lessonId: string; onTake(): void }) {
  const status = statusOf(progressFile, lessonId)
  const quiz = hasQuiz(lessonId)
  return (
    <section className={`record is-${status}`} aria-labelledby="record-title">
      <h3 id="record-title">
        <ClipboardCheck size={15} aria-hidden="true" /> 진행 상태
      </h3>
      <div className="record-row">
        <p className="record-status">
          <span className="record-dot" aria-hidden="true" />
          {statusText(status, passedOn(progressFile, lessonId), attemptsOf(progressFile, lessonId).length, quiz)}
        </p>
        {quiz && (
          <button type="button" className="text-button record-quiz" onClick={onTake}>
            <ListChecks size={15} aria-hidden="true" /> 퀴즈 풀기
          </button>
        )}
      </div>
    </section>
  )
}

// memo: hovering the graph re-renders the app, and the lesson body is costly to render again
export const LessonPanel = memo(function LessonPanel({ curriculum: c, progressFile, lessonId, onClose, onSelect }: Props) {
  const lesson = c.byId.get(lessonId)!
  const branch = c.branches.find((b) => b.id === lesson.branch)!
  const status = statusOf(progressFile, lessonId)
  const missing = lesson.requires.filter((r) => statusOf(progressFile, r) !== 'done')
  const unlocks = c.unlocks.get(lessonId) ?? []
  const [body, setBody] = useState<Root | null>(null)
  const [failed, setFailed] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const quiz = useQuizNav(lessonId)
  const attempts = attemptsOf(progressFile, lessonId)

  // editing a lesson in dev re-runs curriculum/load.ts through HMR, which hands down a new curriculum; the body is
  // fetched again then, so the open panel shows the edit without a reload
  useEffect(() => {
    let live = true
    loadBody(lessonId).then(
      (text) => live && setBody(text),
      () => live && setFailed(true),
    )
    return () => {
      live = false
    }
  }, [lessonId, c])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 })
    headingRef.current?.focus({ preventScroll: true })
  }, [lessonId])

  return (
    <aside className={`panel ${quiz.back ? 'has-quiz-return' : ''}`} aria-labelledby="panel-title" style={{ ['--branch' as string]: branch.color }}>
      <div className="panel-scroll" ref={scrollRef}>
        <header className="panel-head">
          <div className="panel-topline">
            <span className="branch-tag">
              <span className="branch-dot" aria-hidden="true" />
              {branch.title}
            </span>
            <button type="button" className="icon-button" onClick={onClose} aria-label="강의 닫기">
              <X size={20} aria-hidden="true" />
            </button>
          </div>
          <h2 id="panel-title" ref={headingRef} tabIndex={-1}>
            {lesson.title}
          </h2>
          <p className="panel-summary">{lesson.summary}</p>
          <p className="panel-meta">
            <Clock size={15} aria-hidden="true" />
            {formatHours(lesson.hours)}
          </p>

          {lesson.video && (
            <div className="lesson-video">
              <iframe
                src={CONFIG.lesson.videoEmbed + lesson.video}
                title={`${lesson.title} 논문 해설 영상`}
                loading="lazy"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerPolicy="strict-origin-when-cross-origin"
                allowFullScreen
              />
            </div>
          )}

          <StatusCard
            progressFile={progressFile}
            lessonId={lessonId}
            onTake={quiz.take}
          />

          {missing.length > 0 && status === 'todo' && (
            <p className="soft-note">
              아직 통과하지 않은 선수 강의가 {missing.length}개 있습니다. 바로 읽어도 되고, 낯선 용어가 나오면 아래 선수 강의를
              먼저 보면 됩니다.
            </p>
          )}

          {lesson.requires.length > 0 && (
            <section className="relations">
              <h3>선수 강의</h3>
              <ul>
                {lesson.requires.map((id) => (
                  <LessonChip key={id} c={c} p={progressFile} id={id} onSelect={onSelect} />
                ))}
              </ul>
            </section>
          )}
          {unlocks.length > 0 && (
            <section className="relations">
              <h3>이어지는 강의</h3>
              <ul>
                {unlocks.map((id) => (
                  <LessonChip key={id} c={c} p={progressFile} id={id} onSelect={onSelect} />
                ))}
              </ul>
            </section>
          )}

          {lesson.papers.length > 0 && (
            <section className="papers">
              <h3>
                <BookOpen size={15} aria-hidden="true" /> 논문·자료
              </h3>
              <ul>
                {lesson.papers.map((paper) => (
                  <li key={paper.title}>
                    <span className="paper-year">{paper.year}</span>
                    <span>
                      {paper.url ? (
                        <a href={paper.url} target="_blank" rel="noreferrer">
                          {paper.title}
                        </a>
                      ) : (
                        paper.title
                      )}
                      <span className="paper-authors"> — {paper.authors}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </header>

        {body !== null ? (
          <Markdown lessonId={lessonId} tree={body} />
        ) : failed ? (
          <p className="soft-note" role="alert">
            강의를 불러오지 못했습니다. 새로고침하거나 아래 GitHub 링크에서 읽을 수 있습니다.
          </p>
        ) : (
          <div className="body-loading" aria-busy="true" aria-label="강의 불러오는 중">
            <span />
            <span />
            <span />
          </div>
        )}

        <div className="panel-quiz">
          <QuizSection lessonId={lessonId} title={lesson.title} attempts={attempts} nav={quiz} />
        </div>

        <footer className="panel-foot">
          <a href={`${CONFIG.repoUrl}/blob/${CONFIG.repoBranch}/${lesson.path}`} target="_blank" rel="noreferrer">
            GitHub 에서 보기 <ExternalLink size={14} aria-hidden="true" />
          </a>
        </footer>
      </div>
      <QuizReturn attempts={attempts} nav={quiz} />
    </aside>
  )
})
