import { readdirSync, readFileSync, existsSync } from 'node:fs'
import GithubSlugger from 'github-slugger'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { CONFIG } from '../src/config'
import { boxesOverlap, computeLayout, treeCurriculum, type NodePos, type Point } from '../src/curriculum/layout'
import { orderLayers } from '../src/curriculum/order'
import { buildCurriculum } from '../src/curriculum/graph'
import { parseBranches, parseLesson } from '../src/curriculum/parse'

const ROOT = join(__dirname, '..', '..', 'lessons')
// a lecture opens with its goals and closes with a summary and references; the sections between are free
const REQUIRED_SECTIONS = ['## 학습 목표', '## 정리', '## 참고 자료']
// a lesson is a lecture, its figures and its quiz; anything else (code, environments) does not belong
const LESSON_FILES = new Set(['README.md', 'quiz.yaml', 'figures'])

function load() {
  const branches = parseBranches(readFileSync(join(ROOT, 'branches.yaml'), 'utf8'))
  const dirs = readdirSync(ROOT, { withFileTypes: true }).filter((d) => d.isDirectory())
  const lessons = dirs.map((d) => parseLesson(d.name, readFileSync(join(ROOT, d.name, 'README.md'), 'utf8')))
  return buildCurriculum(branches, lessons)
}

describe('curriculum', () => {
  const c = load()

  it('every lesson folder has a README.md', () => {
    for (const p of c.lessons) expect(existsSync(join(ROOT, p.id, 'README.md'))).toBe(true)
  })

  it('every lecture has the standard sections in order', () => {
    for (const p of c.lessons) {
      const positions = REQUIRED_SECTIONS.map((h) => p.body.indexOf(`\n${h}\n`))
      expect(positions.filter((i) => i < 0).length, `${p.path} missing sections`).toBe(0)
      expect([...positions].sort((a, b) => a - b), `${p.path} section order`).toEqual(positions)
    }
  })

  it('every lesson holds only its lecture, figures and quiz', () => {
    for (const p of c.lessons) {
      for (const f of readdirSync(join(ROOT, p.id))) expect(LESSON_FILES.has(f), `${p.id}/${f}`).toBe(true)
    }
  })

  it('lists every lesson\'s papers oldest first', () => {
    for (const p of c.lessons) {
      const years = p.papers.map((q) => q.year)
      expect(years, p.id).toEqual([...years].sort((a, b) => a - b))
    }
  })

  it('gives each explainer video to one lesson only', () => {
    const videos = c.lessons.flatMap((p) => (p.video ? [p.video] : []))
    expect(videos.length).toBeGreaterThan(0)
    expect(new Set(videos).size, 'a video is attached to two lessons').toBe(videos.length)
  })

  it('every lesson ends with a quiz', () => {
    for (const p of c.lessons) expect(existsSync(join(ROOT, p.id, 'quiz.yaml')), `${p.id}/quiz.yaml`).toBe(true)
  })

  it('holds no control characters, which a script writing "\\times" as "\\t" leaves in a formula', () => {
    for (const p of c.lessons) {
      for (const file of ['README.md', 'quiz.yaml']) {
        const bad = readFileSync(join(ROOT, p.id, file), 'utf8').match(/[\x00-\x08\x0b-\x1f]/)
        expect(bad, `${p.id}/${file} has control character U+${bad?.[0].charCodeAt(0).toString(16).padStart(4, '0')}`).toBeNull()
      }
    }
  })

  it('writes ranges with spaced tildes, since GitHub strikes through text between two tight ones', () => {
    // code, math and URLs keep their own tildes
    const protectedSpans = /```[\s\S]*?```|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]*\$|\]\([^)\s]*\)|https?:\/\/\S+/g
    for (const p of c.lessons) {
      for (const file of ['README.md', 'quiz.yaml']) {
        const prose = readFileSync(join(ROOT, p.id, file), 'utf8').replace(protectedSpans, '')
        const tight = prose.match(/[^\s~]~[^\s~]/)
        expect(tight, `${p.id}/${file} has a tight tilde "${tight?.[0]}"; write "A ~ B"`).toBeNull()
      }
    }
  })

  it('section links point at a heading of the same lecture', () => {
    for (const p of c.lessons) {
      const slugger = new GithubSlugger()
      const slugs = new Set([...p.body.matchAll(/^#{1,6} (.+)$/gm)].map((m) => slugger.slug(m[1])))
      for (const m of p.body.matchAll(/\]\(#([^)]+)\)/g)) {
        expect(slugs.has(decodeURIComponent(m[1])), `${p.path} links to missing section #${m[1]}`).toBe(true)
      }
    }
  })

  it('links into another lecture point at a heading it has', () => {
    const slugsOf = new Map(
      c.lessons.map((p) => {
        const slugger = new GithubSlugger()
        return [p.id, new Set([...p.body.matchAll(/^#{1,6} (.+)$/gm)].map((m) => slugger.slug(m[1])))] as const
      }),
    )
    for (const p of c.lessons) {
      for (const m of p.body.matchAll(/\]\(\.\.\/([a-z0-9-]+)\/README\.md#([^)]+)\)/g)) {
        const target = slugsOf.get(m[1])
        if (!target) continue // a missing lesson is reported by the lesson-link test
        expect(target.has(decodeURIComponent(m[2])), `${p.path} links to missing section ${m[1]}#${m[2]}`).toBe(true)
      }
    }
  })

  it('every progress file belongs to a lesson', () => {
    const dir = join(ROOT, '..', CONFIG.quiz.progressDir)
    if (!existsSync(dir)) return
    for (const f of readdirSync(dir)) expect(c.byId.has(f.replace(/\.json$/, '')), `${CONFIG.quiz.progressDir}/${f}`).toBe(true)
  })

  it('links to other lessons point at existing lessons', () => {
    for (const p of c.lessons) {
      for (const m of p.body.matchAll(/\]\(\.\.\/([a-z0-9-]+)\/README\.md\)/g)) {
        expect(c.byId.has(m[1]), `${p.path} links to missing lesson ${m[1]}`).toBe(true)
      }
    }
  })

  it('has exactly one starting point', () => {
    expect(c.lessons.filter((p) => p.requires.length === 0).map((p) => p.id)).toEqual(['vectors-matrices'])
  })

  it('lays out every lesson without overlapping labels', async () => {
    const layout = await computeLayout(c)
    expect(layout.nodes.size).toBe(c.lessons.length)
    const nodes = [...layout.nodes.values()]
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        expect(boxesOverlap(nodes[i], nodes[j]), `${nodes[i].id} overlaps ${nodes[j].id}`).toBe(false)
      }
    }
  })

  it('puts the starting lesson at the centre', async () => {
    const layout = await computeLayout(c)
    const start = c.lessons.find((p) => p.requires.length === 0)!
    expect(layout.nodes.get(start.id)).toMatchObject({ x: 0, y: 0 })
  })

  // distance from the centre along the lesson's tree direction
  const reach = (n: NodePos) => {
    const t = c.trees.find((t) => t.id === n.tree)
    if (!t) return 0
    const a = (t.angle * Math.PI) / 180
    return n.x * Math.sin(a) - n.y * Math.cos(a)
  }

  it('grows every tree out of the centre in its own direction, prerequisites first', async () => {
    const layout = await computeLayout(c)
    for (const p of c.lessons) {
      if (p.requires.length === 0 || c.hub.has(p.id)) continue
      const n = layout.nodes.get(p.id)!
      expect(n.tree, p.id).toBe(c.trees.find((t) => t.branches.includes(p.branch))!.id)
      expect(reach(n), p.id).toBeGreaterThan(0)
      for (const r of p.requires) {
        const q = layout.nodes.get(r)!
        if (q.tree === n.tree) expect(reach(q), `${r} → ${p.id}`).toBeLessThan(reach(n))
      }
    }
  })

  it('keeps the core chain in one straight line', async () => {
    const layout = await computeLayout(c)
    const xs = [...layout.nodes.values()].filter((n) => n.core).map((n) => n.x)
    expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(1)
  })

  it('names every tree inside its own area, off the hub', async () => {
    const layout = await computeLayout(c)
    expect(layout.sectors.map((r) => r.tree).sort()).toEqual(c.trees.map((t) => t.id).sort())
    expect(layout.dividers).toHaveLength(c.trees.length)
    for (const r of layout.sectors) {
      for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        const [x, y] = [r.x + (dx * r.width) / 2, r.y + (dy * r.height) / 2]
        const deg = (((Math.atan2(x, -y) * 180) / Math.PI - r.start) % 360 + 360) % 360
        expect(deg, `${r.tree} name corner`).toBeLessThan(r.end - r.start)
      }
      expect(Math.hypot(Math.max(Math.abs(r.x) - r.width / 2, 0), Math.max(Math.abs(r.y) - r.height / 2, 0)), r.tree).toBeGreaterThan(layout.rim)
    }
  })

  it('centres each tree name on its picture, and the picture on the tree lessons', async () => {
    const layout = await computeLayout(c)
    for (const r of layout.sectors) {
      expect([r.x, r.y], r.tree).toEqual([r.art.x, r.art.y])
      for (const n of [...layout.nodes.values()].filter((n) => n.tree === r.tree)) {
        expect(Math.abs(n.x - r.art.x), `${n.id} inside ${r.tree} picture`).toBeLessThanOrEqual(r.art.size / 2)
        expect(Math.abs(n.y - r.art.y), `${n.id} inside ${r.tree} picture`).toBeLessThanOrEqual(r.art.size / 2)
      }
    }
  })

  it('sets every hub lesson on its ring, and every tree outside the rim', async () => {
    const layout = await computeLayout(c)
    const outer = layout.rim
    for (const n of layout.nodes.values()) {
      const r = Math.hypot(n.x, n.y)
      const ring = c.hub.get(n.id)
      if (ring === undefined) expect(r, n.id).toBeGreaterThan(outer)
      else if (ring > 0) expect(r, n.id).toBeCloseTo(layout.rings[ring - 1], 0)
    }
  })

  it('routes every edge inside a tree around the lessons it does not connect', async () => {
    const layout = await computeLayout(c)
    for (const e of layout.edges) {
      // an edge from the hub or another tree is not routed through the tree's layers
      if (layout.nodes.get(e.from)!.tree !== layout.nodes.get(e.to)!.tree) continue
      for (const n of layout.nodes.values()) {
        if (n.id === e.from || n.id === e.to) continue
        // sample each segment so a corner cut between two points counts too
        for (let i = 1; i < e.points.length; i++) {
          const a = e.points[i - 1]
          const b = e.points[i]
          for (let t = 0; t <= 1; t += 0.25) {
            const x = a.x + (b.x - a.x) * t - n.x
            const y = a.y + (b.y - a.y) * t - n.y
            const inside = x > n.box.left && x < n.box.right && y > n.box.top && y < n.box.bottom
            expect(inside, `${e.from} → ${e.to} crosses ${n.id}`).toBe(false)
          }
        }
      }
    }
  })

  it('ends each tree with its outer branch', () => {
    const L = CONFIG.layout
    const outer = new Set(c.branches.filter((b) => b.outer).map((b) => b.id))
    for (const t of c.trees) {
      const { curriculum } = treeCurriculum(c, t)
      const { rows } = orderLayers(curriculum, { restarts: 1, sweeps: L.orderSweeps, seed: L.orderSeed })
      const last = rows.length - 1
      const rowOf = (id: string) => rows.findIndex((row) => row.includes(id))
      for (const p of curriculum.lessons.filter((p) => outer.has(p.branch))) expect(rowOf(p.id), p.id).toBe(last)
    }
  })

  it('draws no more crossings inside a tree than the layer order it was given', async () => {
    const L = CONFIG.layout
    const layout = await computeLayout(c)
    const side = (a: Point, b: Point, p: Point) => (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
    const cross = (a: Point, b: Point, p: Point, q: Point) => side(a, b, p) * side(a, b, q) < 0 && side(p, q, a) * side(p, q, b) < 0
    for (const t of c.trees) {
      const { curriculum } = treeCurriculum(c, t)
      const order = orderLayers(curriculum, { restarts: L.orderRestarts, sweeps: L.orderSweeps, seed: L.orderSeed })
      const own = layout.edges.filter((e) => curriculum.byId.get(e.to)?.requires.includes(e.from))
      // count every place two edges meet, as orderLayers counts one crossing per layer gap; edges gathered into one line
      // cross another edge at one place
      const drawn = new Set<string>()
      own.forEach((e, i) => {
        for (const g of own.slice(i + 1)) {
          // edges that share a lesson meet at its port; that is not a crossing
          if ([e.from, e.to].some((id) => id === g.from || id === g.to)) continue
          e.points.slice(1).forEach((b, k) => {
            g.points.slice(1).forEach((q, m) => {
              const a = e.points[k]
              if (cross(a, b, g.points[m], q)) {
                const s = side(g.points[m], q, a) / (side(g.points[m], q, a) - side(g.points[m], q, b))
                drawn.add(`${Math.round(a.x + s * (b.x - a.x))},${Math.round(a.y + s * (b.y - a.y))}`)
              }
            })
          })
        }
      })
      expect(drawn.size, t.id).toBeLessThanOrEqual(order.crossings)
    }
  })
})
