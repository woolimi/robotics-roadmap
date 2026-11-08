import { parse as parseYaml } from 'yaml'
import { CONFIG } from '../config.ts'
import { fail } from './graph.ts'
import type { Topic, Variant } from './quiz.ts'
import type { Branch, Hub, MapSpec, Paper, Lesson, Tree } from './types.ts'

// Readers of the repo's YAML: branches.yaml, lesson frontmatter and quiz.yaml. The server (lesson-data.ts, quiz-server.ts)
// and the tests use them; the page gets their results already parsed, so it carries no YAML parser.

const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/

function str(value: unknown, where: string, field: string): string {
  if (typeof value !== 'string' || value.trim() === '') fail(where, `"${field}" must be a non-empty string`)
  return value.trim()
}

export function parseBranches(source: string): MapSpec {
  const data = parseYaml(source) as Record<string, unknown> | null
  if (!data || !Array.isArray(data.trees) || !Array.isArray(data.branches)) fail('branches.yaml', 'expected "trees" and "branches" lists')
  const rawRings = ((data.hub ?? {}) as Record<string, unknown>).rings ?? []
  if (!Array.isArray(rawRings)) fail('branches.yaml', '"hub.rings" must be a list')
  const hub: Hub = {
    rings: rawRings.map((raw, i) => {
      const ring = (raw ?? {}) as Record<string, unknown>
      const ids = (field: string) => {
        const v = ring[field] ?? []
        if (!Array.isArray(v) || v.some((x) => typeof x !== 'string')) fail('branches.yaml', `"hub.rings[${i}].${field}" must be a list of ids`)
        return v as string[]
      }
      return { branches: ids('branches'), lessons: ids('lessons') }
    }),
  }
  const trees: Tree[] = data.trees.map((raw, i) => {
    const where = `branches.yaml trees[${i}]`
    const t = raw as Record<string, unknown>
    if (typeof t.angle !== 'number' || t.angle < 0 || t.angle >= 360) fail(where, '"angle" must be a number in [0, 360)')
    if (!Array.isArray(t.branches) || t.branches.some((b) => typeof b !== 'string')) fail(where, '"branches" must be a list of branch ids')
    return { id: str(t.id, where, 'id'), title: str(t.title, where, 'title'), color: str(t.color, where, 'color'), angle: t.angle, branches: t.branches as string[] }
  })
  const branches: Branch[] = data.branches.map((raw, i) => {
    const where = `branches.yaml branches[${i}]`
    const b = raw as Record<string, unknown>
    if (b.core !== undefined && typeof b.core !== 'boolean') fail(where, '"core" must be true or false')
    if (b.outer !== undefined && typeof b.outer !== 'boolean') fail(where, '"outer" must be true or false')
    return {
      id: str(b.id, where, 'id'),
      title: str(b.title, where, 'title'),
      summary: str(b.summary, where, 'summary'),
      color: str(b.color, where, 'color'),
      core: b.core === true,
      outer: b.outer === true,
    }
  })
  return { hub, trees, branches }
}

function parsePapers(value: unknown, where: string): Paper[] {
  if (value === undefined) return []
  if (!Array.isArray(value)) fail(where, '"papers" must be a list')
  return value.map((raw, i) => {
    const p = raw as Record<string, unknown>
    const w = `${where} papers[${i}]`
    if (typeof p.year !== 'number') fail(w, '"year" must be a number')
    return {
      title: str(p.title, w, 'title'),
      authors: str(p.authors, w, 'authors'),
      year: p.year,
      url: p.url === undefined ? undefined : str(p.url, w, 'url'),
    }
  })
}

// a YouTube watch URL, the form a reader can open from the frontmatter on GitHub
const YOUTUBE_WATCH = /^https:\/\/www\.youtube\.com\/watch\?v=([A-Za-z0-9_-]{11})$/

function parseVideo(value: unknown, where: string): string | undefined {
  if (value === undefined) return undefined
  const id = YOUTUBE_WATCH.exec(str(value, where, 'video'))?.[1]
  if (!id) fail(where, '"video" must be a URL like https://www.youtube.com/watch?v=<11-character id>')
  return id
}

/** A lesson README without its frontmatter. */
export const bodyOf = (source: string) => source.replace(FRONTMATTER, '')

/** Parse one `lessons/<id>/README.md`. `dir` is the folder name, which must equal the id. */
export function parseLesson(dir: string, source: string): Lesson {
  const path = `lessons/${dir}/README.md`
  const match = FRONTMATTER.exec(source)
  if (!match) fail(path, 'missing YAML frontmatter')
  const fm = (parseYaml(match[1]) ?? {}) as Record<string, unknown>

  const id = str(fm.id, path, 'id')
  if (id !== dir) fail(path, `id "${id}" must match folder name "${dir}"`)
  const requires = fm.requires ?? []
  if (!Array.isArray(requires) || requires.some((r) => typeof r !== 'string')) {
    fail(path, '"requires" must be a list of lesson ids')
  }
  if (typeof fm.hours !== 'number' || fm.hours <= 0) fail(path, '"hours" must be a positive number')

  return {
    id,
    title: str(fm.title, path, 'title'),
    branch: str(fm.branch, path, 'branch'),
    requires: requires as string[],
    hours: fm.hours,
    summary: str(fm.summary, path, 'summary'),
    papers: parsePapers(fm.papers, path),
    video: parseVideo(fm.video, path),
    body: source.slice(match[0].length),
    path,
  }
}

// quiz.yaml

const TOPIC_ID = /^[a-z0-9-]+$/
// a GitHub heading slug: lower-case letters of any script, digits, `-` and `_`
const SECTION_ID = /^[\p{Ll}\p{Lo}\p{Lm}\p{M}\p{N}_-]+$/u

function section(value: unknown, where: string): string {
  const slug = text(value, where)
  if (!SECTION_ID.test(slug)) throw new Error(`${where}: 제목 slug 가 아닙니다`)
  return slug
}

function text(value: unknown, where: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${where}: 문자열이 아닙니다`)
  return value.trim()
}

/** Parse lessons/<id>/quiz.yaml. Throws on a malformed bank so a broken file shows up in tests, not mid-quiz. */
export function parseBank(source: string): Topic[] {
  const doc: unknown = parseYaml(source)
  if (!Array.isArray(doc) || doc.length === 0) throw new Error('quiz.yaml: topic 목록이 비었습니다')
  const seen = new Set<string>()
  return doc.map((raw, i) => {
    const t = raw as Record<string, unknown>
    const topic = text(t?.topic, `topic ${i}`)
    if (!TOPIC_ID.test(topic) || seen.has(topic)) throw new Error(`topic ${topic}: id 가 잘못됐거나 중복입니다`)
    seen.add(topic)
    // a retake asks the next variant, so one variant would repeat the same question
    if (!Array.isArray(t.variants) || t.variants.length < 2) throw new Error(`${topic}: variant 가 2개 이상이어야 합니다`)
    const variants = t.variants.map((rawV, j): Variant => {
      const v = rawV as Record<string, unknown>
      const where = `${topic}.variants[${j}]`
      const choices = Array.isArray(v?.choices) ? v.choices.map((c, k) => text(c, `${where}.choices[${k}]`)) : []
      if (choices.length !== CONFIG.quiz.choiceCount) throw new Error(`${where}: 보기는 ${CONFIG.quiz.choiceCount}개여야 합니다`)
      const answer = v.answer
      if (typeof answer !== 'number' || !Number.isInteger(answer) || answer < 0 || answer >= choices.length) {
        throw new Error(`${where}: answer 가 보기 범위를 벗어났습니다`)
      }
      return {
        question: text(v.question, `${where}.question`),
        choices,
        answer,
        explanation: text(v.explanation, `${where}.explanation`),
        ...(v.section !== undefined && { section: section(v.section, `${where}.section`) }),
      }
    })
    return { topic, title: text(t.title, `${topic}.title`), section: section(t.section, `${topic}.section`), variants }
  })
}
