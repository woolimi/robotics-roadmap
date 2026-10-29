import type { Root } from 'hast'
import { BookOpen, Check, ChevronRight, CornerUpLeft, History, ListChecks, X } from './icons'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { CONFIG } from '../config'
import { keepOpenOnEscape } from './dialog'
import { hasQuiz, loadBank } from '../curriculum/load'
import { progress } from '../curriculum/progress'
import {
  clearedAt,
  grade,
  passed,
  pickVariants,
  sectionOf,
  wrongTopics,
  type Answer,
  type Attempt,
  type Pick,
  type Topic,
} from '../curriculum/quiz'
import { Markdown, scrollToSettled, sectionElement } from './Markdown'

const url = (id: string) => `${CONFIG.quiz.endpoint}/${id}`

async function request<T>(id: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url(id), init)
  const body = res.headers.get('Content-Type')?.includes('application/json') ? await res.json() : null
  if (!body) throw new Error('offline')
  if (!res.ok) throw new Error(body.error ?? `요청 실패 (${res.status})`)
  return body as T
}

const score = (a: Attempt) => `${a.answers.filter((x) => x.correct).length} / ${a.answers.length}`

type View = { kind: 'take'; run: number } | { kind: 'review'; index: number; focus?: string }

/** where "강의에서 읽기" left the review, so the bar under the lecture can bring the learner back to it */
interface Back {
  index: number
  topic: string
}

/**
 * The quiz dialog and the trip between a missed question and the lecture section that teaches it. Lives in the lesson
 * panel because the way back is drawn over the lecture, outside the quiz card.
 */
export function useQuizNav(lessonId: string) {
  const [owner, setOwner] = useState(lessonId)
  const [view, setView] = useState<View | null>(null)
  const [back, setBack] = useState<Back | null>(null)
  // `${attempt index}:${topic}` of the explanations whose section was opened while this lesson stayed open
  const [read, setRead] = useState<ReadonlySet<string>>(new Set())
  const [target, setTarget] = useState<string | null>(null)
  const runs = useRef(0)

  // another lesson in the same panel starts with no dialog and no way back
  if (owner !== lessonId) {
    setOwner(lessonId)
    setView(null)
    setBack(null)
    setRead(new Set())
    setTarget(null)
  }

  // scroll once the dialog is gone: the modal makes the lecture inert, so the heading could not take focus before
  useEffect(() => {
    if (!target) return
    setTarget(null)
    const heading = sectionElement(target)
    if (!heading) return
    heading.tabIndex = -1
    heading.focus({ preventScroll: true })
    // light the heading once the figures above it have settled and the scroll has started, so it is seen arriving
    void scrollToSettled(heading).then(() => {
      heading.classList.remove('is-quiz-target')
      void heading.offsetWidth // restart the highlight when the same heading is opened twice in a row
      heading.classList.add('is-quiz-target')
      heading.addEventListener('animationend', () => heading.classList.remove('is-quiz-target'), { once: true })
    })
  }, [target])

  return {
    view,
    back,
    read,
    take: () => {
      setBack(null)
      setView({ kind: 'take', run: ++runs.current })
    },
    review: (index: number, focus?: string) => {
      setBack(null)
      setView({ kind: 'review', index, focus })
    },
    close: () => setView(null),
    jump: (index: number, topic: string, section: string) => {
      setView(null)
      setBack({ index, topic })
      setRead((prev) => new Set(prev).add(`${index}:${topic}`))
      setTarget(section)
    },
    dismissBack: () => setBack(null),
  }
}

export type QuizNav = ReturnType<typeof useQuizNav>

interface QuizProps {
  lessonId: string
  title: string
  attempts: readonly Attempt[]
  nav: QuizNav
}

/** Quiz card at the end of a lecture, with the dialog it opens. Renders nothing for lessons without a bank. */
export function QuizSection({ lessonId, title, attempts, nav }: QuizProps) {
  if (!hasQuiz(lessonId)) return null
  const last = attempts.at(-1)
  const cleared = clearedAt(attempts) >= 0
  const wrong = cleared ? 0 : wrongTopics(attempts).length
  const { view } = nav

  return (
    <section className="quiz-card" aria-labelledby="quiz-card-title">
      <h3 id="quiz-card-title">
        <ListChecks size={15} aria-hidden="true" /> 퀴즈
      </h3>
      <p className="quiz-card-text">
        강의의 핵심을 한 문항씩 묻습니다. 틀린 문항은 해설과 함께 그 내용을 다룬 절로 가는 길을 보여 주고, 틀린 문항만 같은 원리의 다른
        문항으로 다시 풉니다. 틀린 문항이 남지 않으면 이 강의를 통과합니다.
      </p>
      {last && (
        <p className={`quiz-card-last ${cleared ? 'is-pass' : 'is-fail'}`}>
          마지막 응시 {last.date} · {score(last)} 정답{wrong > 0 && ` · 남은 문항 ${wrong}개`}
        </p>
      )}
      {!import.meta.env.DEV && (
        <p className="quiz-card-text">
          이 사이트에서는 풀어 볼 수만 있고 결과는 새로고침하면 사라집니다. 기록을 남기려면 리포를 fork 해 <code>site/</code> 에서{' '}
          <code>npm run dev</code> 로 띄웁니다.
        </p>
      )}
      <div className="quiz-card-actions">
        <button type="button" className="primary-button quiz-start" onClick={nav.take}>
          {!last ? '퀴즈 풀기' : wrong > 0 ? `틀린 문제 다시 풀기 (${wrong}문항)` : '처음부터 다시 풀기'}
        </button>
        {last && (
          <button type="button" className="text-button quiz-history" onClick={() => nav.review(attempts.length - 1)}>
            <History size={15} aria-hidden="true" /> 해설과 지난 기록 ({attempts.length}회)
          </button>
        )}
      </div>
      {view?.kind === 'take' && (
        <TakeDialog
          key={view.run}
          lessonId={lessonId}
          title={title}
          attempts={attempts}
          onClose={nav.close}
          onSubmitted={(attempt) => {
            progress.add(lessonId, attempt)
            nav.review(attempts.length)
          }}
        />
      )}
      {view?.kind === 'review' && (
        <ReviewDialog
          lessonId={lessonId}
          title={title}
          attempts={attempts}
          index={view.index}
          focus={view.focus}
          read={nav.read}
          onPick={(i) => nav.review(i)}
          onJump={nav.jump}
          onRetry={nav.take}
          onClose={nav.close}
        />
      )}
    </section>
  )
}

/** The way back to the review after "강의에서 읽기", pinned over the bottom of the lecture. */
export function QuizReturn({ attempts, nav }: { attempts: readonly Attempt[]; nav: QuizNav }) {
  const attempt = nav.back && attempts[nav.back.index]
  if (!nav.back || !attempt) return null
  const { index, topic } = nav.back
  const wrong = attempt.answers.filter((a) => !a.correct)
  const read = wrong.filter((a) => nav.read.has(`${index}:${a.topic}`)).length
  return (
    <div className="quiz-return" role="region" aria-label="퀴즈 해설로 돌아가기">
      <button type="button" className="quiz-return-main" onClick={() => nav.review(index, topic)}>
        <CornerUpLeft size={18} aria-hidden="true" />
        <span>
          <strong>퀴즈 해설로 돌아가기</strong>
          {wrong.length > 0 && (
            <span className="quiz-return-meta">
              틀린 문항 {wrong.length}개 중 {read}개 읽음
            </span>
          )}
        </span>
      </button>
      <button type="button" className="icon-button" onClick={nav.dismissBack} aria-label="돌아가기 표시 닫기">
        <X size={18} aria-hidden="true" />
      </button>
    </div>
  )
}

/** The lesson's question bank, fetched when a dialog opens. */
function useBank(lessonId: string) {
  const [bank, setBank] = useState<Topic<Root>[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    loadBank(lessonId).then(
      (b) => live && setBank(b),
      (e) => live && setError(`문항을 불러오지 못했습니다: ${e instanceof Error ? e.message : e}`),
    )
    return () => {
      live = false
    }
  }, [lessonId])
  return { bank, error }
}

/** Modal shell shared by taking a quiz and reading its review: kicker, close button, scrolling body, footer. */
function QuizDialog({ kicker, onClose, top, children, foot }: { kicker: string; onClose(): void; top?: ReactNode; children: ReactNode; foot: ReactNode }) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    dialogRef.current?.showModal()
  }, [])
  return (
    <dialog
      ref={dialogRef}
      className="quiz-dialog"
      aria-labelledby="quiz-heading"
      onKeyDown={keepOpenOnEscape}
      onCancel={(e) => e.preventDefault()}
    >
      <header className="quiz-head">
        <p className="quiz-kicker">{kicker}</p>
        <button type="button" className="icon-button" onClick={onClose} aria-label="퀴즈 닫기">
          <X size={20} aria-hidden="true" />
        </button>
        {top}
      </header>
      <div className="quiz-body">{children}</div>
      <footer className="quiz-foot">{foot}</footer>
    </dialog>
  )
}

interface TakeProps {
  lessonId: string
  title: string
  attempts: readonly Attempt[]
  onClose(): void
  onSubmitted(attempt: Attempt): void
}

function TakeDialog({ lessonId, title, attempts: opened, onClose, onSubmitted }: TakeProps) {
  // the questions are fixed when the dialog opens; the submission below must not re-pick them
  const [attempts] = useState(opened)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const { bank, error: loadError } = useBank(lessonId)
  const [picks, setPicks] = useState<Pick[]>([])
  const [choices, setChoices] = useState<(number | null)[]>([])
  const [step, setStep] = useState(0)
  const [sending, setSending] = useState(false)
  const [sendError, setSendError] = useState<string | null>(null)
  const error = loadError ?? sendError

  useEffect(() => {
    if (!bank) return
    const next = pickVariants(bank, attempts)
    setPicks(next)
    setChoices(next.map(() => null))
  }, [bank, attempts])

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true })
    headingRef.current?.closest('.quiz-body')?.scrollTo({ top: 0 })
  }, [step, picks])

  const total = picks.length
  const topic = bank?.find((t) => t.topic === picks[step]?.topic)
  const variant = topic ? topic.variants[picks[step].variant] : undefined
  const chosen = choices[step]
  const isLast = step === total - 1

  async function submit() {
    setSending(true)
    setSendError(null)
    try {
      // a built site has no dev server: grade in the page and keep the result only until reload
      if (!import.meta.env.DEV) {
        onSubmitted({ date: new Date().toISOString().slice(0, 10), answers: grade(bank!, picks, choices as number[]) })
        return
      }
      const attempt = await request<Attempt>(lessonId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: picks.map((p, i) => ({ ...p, choice: choices[i] })) }),
      })
      onSubmitted(attempt)
    } catch (e) {
      setSendError(e instanceof Error && e.message !== 'offline' ? e.message : '제출하지 못했습니다. dev server 가 켜져 있는지 확인합니다')
      setSending(false)
    }
  }

  return (
    <QuizDialog
      kicker={`퀴즈 · ${title}`}
      onClose={onClose}
      top={
        total > 0 && (
          <div className="quiz-progress" aria-hidden="true">
            <span style={{ width: `${((step + 1) / total) * 100}%` }} />
          </div>
        )
      }
      foot={
        <>
          <button type="button" className="text-button" onClick={() => setStep(step - 1)} disabled={step === 0 || sending}>
            이전
          </button>
          {isLast ? (
            <button type="button" className="primary-button" onClick={submit} disabled={chosen == null || sending}>
              {sending ? '제출 중…' : '제출'}
            </button>
          ) : (
            <button type="button" className="primary-button" onClick={() => setStep(step + 1)} disabled={chosen == null || !variant}>
              다음
            </button>
          )}
        </>
      }
    >
      {topic && variant ? (
        <fieldset className="quiz-question">
          <legend className="sr-only">
            {step + 1}번 문항 / 전체 {total}문항
          </legend>
          <p className="quiz-step">
            {step + 1} / {total} · {topic.title}
          </p>
          <h2 id="quiz-heading" ref={headingRef} tabIndex={-1} className="sr-only">
            {step + 1}번 문항
          </h2>
          <Markdown lessonId={lessonId} tree={variant.question} />
          <div className="quiz-choices">
            {variant.choices.map((choice, i) => (
              <label key={i} className={`quiz-choice ${chosen === i ? 'is-on' : ''}`}>
                <input
                  type="radio"
                  name={`q-${step}`}
                  checked={chosen === i}
                  onChange={() => setChoices((prev) => prev.map((c, j) => (j === step ? i : c)))}
                />
                <span className="quiz-choice-key" aria-hidden="true">
                  {i + 1}
                </span>
                <Markdown lessonId={lessonId} tree={choice} inline />
              </label>
            ))}
          </div>
        </fieldset>
      ) : (
        !error && (
          <p id="quiz-heading" className="quiz-card-text" aria-busy="true">
            문항 불러오는 중…
          </p>
        )
      )}
      {error && (
        <p className="quiz-error" role="alert">
          {error}
        </p>
      )}
    </QuizDialog>
  )
}

interface ReviewProps {
  lessonId: string
  title: string
  attempts: readonly Attempt[]
  index: number
  /** topic whose explanation the learner left for the lecture; the review opens on it */
  focus?: string
  read: ReadonlySet<string>
  onPick(index: number): void
  onJump(index: number, topic: string, section: string): void
  onRetry(): void
  onClose(): void
}

/** One attempt's answers with explanations, a way into the lecture for each, and a switch to the earlier attempts. */
function ReviewDialog({ lessonId, title, attempts, index, focus, read, onPick, onJump, onRetry, onClose }: ReviewProps) {
  const { bank, error } = useBank(lessonId)
  const bodyRef = useRef<HTMLDivElement>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const attempt = attempts[index]
  const latest = index === attempts.length - 1
  const left = clearedAt(attempts) >= 0 ? 0 : wrongTopics(attempts).length

  useEffect(() => {
    if (!bank || !attempt) return
    const scroller = bodyRef.current?.closest('.quiz-body')
    const item = focus && bodyRef.current?.querySelector<HTMLElement>(`[data-topic="${focus}"]`)
    // scroll only the dialog body: scrollIntoView would also scroll the dialog box and push its header out of view
    if (scroller && item) {
      scroller.scrollTop += item.getBoundingClientRect().top - scroller.getBoundingClientRect().top - CONFIG.quiz.reviewFocusGap
      item.querySelector<HTMLElement>('.quiz-reread')?.focus({ preventScroll: true })
    } else {
      scroller?.scrollTo({ top: 0 })
      headingRef.current?.focus({ preventScroll: true })
    }
  }, [bank, attempt, focus])

  // answers whose question is still in the bank; a topic or variant removed since the attempt has nothing to show
  const rows = (bank && attempt?.answers.flatMap((a) => {
    const topic = bank.find((t) => t.topic === a.topic)
    const variant = topic?.variants[a.variant]
    return topic && variant ? [{ a, topic }] : []
  })) ?? []
  const wrong = rows.filter((r) => !r.a.correct)
  const right = rows.filter((r) => r.a.correct)
  const item = ({ a, topic }: { a: Answer; topic: Topic<Root> }) => (
    <ReviewItem key={a.topic} lessonId={lessonId} answer={a} topic={topic} read={read.has(`${index}:${a.topic}`)} onJump={(s) => onJump(index, a.topic, s)} />
  )

  return (
    <QuizDialog
      kicker={`퀴즈 해설 · ${title}`}
      onClose={onClose}
      foot={
        left > 0 ? (
          <>
            <button type="button" className="text-button" onClick={onClose}>
              닫기
            </button>
            <button type="button" className="primary-button" onClick={onRetry}>
              틀린 문제 다시 풀기 ({left}문항)
            </button>
          </>
        ) : (
          <button type="button" className="primary-button" onClick={onClose}>
            닫기
          </button>
        )
      }
    >
      <div className="quiz-results" ref={bodyRef}>
        {attempts.length > 1 && (
          <div className="quiz-attempts" role="group" aria-label="응시 기록">
            {attempts.map((x, i) => (
              <button
                key={i}
                type="button"
                className={`quiz-attempt ${passed(x.answers) ? 'is-pass' : 'is-fail'}`}
                aria-pressed={i === index}
                title={x.date}
                onClick={() => onPick(i)}
              >
                {i + 1}회 <span className="quiz-attempt-score">{score(x)}</span>
              </button>
            ))}
          </div>
        )}
        {attempt && bank ? (
          <>
            <h2 id="quiz-heading" ref={headingRef} tabIndex={-1} className={passed(attempt.answers) ? 'is-pass' : 'is-fail'}>
              {score(attempt)} 정답
            </h2>
            <p className="quiz-attempt-date">
              {index + 1}회 응시 · {attempt.date}
            </p>
            {latest && <p className="quiz-verdict">{verdict(attempt, clearedAt(attempts) === index)}</p>}
            {wrong.length > 0 && <ol className="quiz-review">{wrong.map(item)}</ol>}
            {right.length > 0 &&
              (wrong.length > 0 ? (
                <details className="quiz-right">
                  <summary>
                    <ChevronRight size={16} aria-hidden="true" /> 맞힌 문항 {right.length}개
                  </summary>
                  <ol className="quiz-review">{right.map(item)}</ol>
                </details>
              ) : (
                <ol className="quiz-review">{right.map(item)}</ol>
              ))}
          </>
        ) : (
          !error && (
            <p id="quiz-heading" className="quiz-card-text" aria-busy="true">
              해설 불러오는 중…
            </p>
          )
        )}
        {error && (
          <p className="quiz-error" role="alert">
            {error}
          </p>
        )}
      </div>
    </QuizDialog>
  )
}

function verdict(attempt: Attempt, clearedHere: boolean): string {
  const wrong = attempt.answers.filter((a) => !a.correct).length
  if (wrong === 0) return clearedHere ? '틀린 문항이 남지 않아 이 강의를 통과했습니다. 지도에서 이어지는 강의가 밝아집니다.' : '모두 맞혔습니다.'
  return `틀린 문항이 ${wrong}개입니다. 해설 아래 "강의에서 읽기" 로 그 내용을 다룬 절을 읽고, 강의 아래쪽에 뜨는 "퀴즈 해설로 돌아가기" 로 돌아와 다음 문항을 봅니다.`
}

function ReviewItem({ lessonId, answer: a, topic, read, onJump }: { lessonId: string; answer: Answer; topic: Topic<Root>; read: boolean; onJump(section: string): void }) {
  const v = topic.variants[a.variant]
  const section = sectionOf(topic, a.variant)
  // the lecture is already on screen behind the dialog; a heading that is not there (body still loading) gets no button
  const heading = sectionElement(section)?.textContent
  return (
    <li className={a.correct ? 'is-correct' : 'is-wrong'} data-topic={a.topic}>
      <p className="quiz-review-title">
        {a.correct ? <Check size={16} aria-hidden="true" /> : <X size={16} aria-hidden="true" />}
        <span className="sr-only">{a.correct ? '정답' : '오답'}</span>
        {topic.title}
      </p>
      <Markdown lessonId={lessonId} tree={v.question} />
      <dl>
        {!a.correct && (
          <>
            <dt>고른 답</dt>
            <dd>
              <Markdown lessonId={lessonId} tree={v.choices[a.choice]} inline />
            </dd>
          </>
        )}
        <dt>정답</dt>
        <dd>
          <Markdown lessonId={lessonId} tree={v.choices[v.answer]} inline />
        </dd>
      </dl>
      <Markdown lessonId={lessonId} tree={v.explanation} />
      {heading && (
        <button type="button" className={`text-button quiz-reread ${read ? 'is-read' : ''}`} onClick={() => onJump(section)}>
          <BookOpen size={15} aria-hidden="true" />
          <span>
            강의에서 읽기 · <span className="quiz-reread-heading">{heading}</span>
          </span>
          {read && (
            <span className="quiz-reread-done">
              <Check size={14} aria-hidden="true" /> 읽음
            </span>
          )}
        </button>
      )}
    </li>
  )
}
