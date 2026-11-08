export interface Branch {
  id: string
  title: string
  summary: string
  color: string
  /** the foundations every other branch builds on; its chain is laid out as the graph's straight trunk */
  core: boolean
  /** the newest research; its lessons sit on the last row of their tree */
  outer: boolean
}

/** One tree of the map: a group of branches that grows from the starting lesson in one direction. */
export interface Tree {
  id: string
  /** the name written in the tree's area of the map */
  title: string
  /** the faint tint of the tree's area */
  color: string
  /** direction the tree grows from the centre of the map, degrees clockwise from up */
  angle: number
  /** its branches, laid side by side in this order, clockwise around the centre */
  branches: string[]
}

/** One ring of the hub: lessons of one kind. */
export interface HubRing {
  /** branches whose lessons all sit on this ring */
  branches: string[]
  /** single lessons of tree branches that sit on this ring */
  lessons: string[]
}

/** The basics at the centre of the map that every tree starts from. */
export interface Hub {
  /** innermost first */
  rings: HubRing[]
}

/** lessons/branches.yaml */
export interface MapSpec {
  hub: Hub
  trees: Tree[]
  branches: Branch[]
}

export interface Paper {
  title: string
  authors: string
  year: number
  url?: string
}

export interface Lesson {
  id: string
  title: string
  branch: string
  requires: string[]
  hours: number
  summary: string
  papers: Paper[]
  /** YouTube id of the explainer video shown at the top of the lesson, from the frontmatter's `video` watch URL */
  video?: string
  /** markdown body without frontmatter */
  body: string
  /** repo-relative path of the subject file */
  path: string
}

export interface Curriculum {
  /** hub lesson id -> its ring, 1 for the innermost; the start sits at the centre, ring 0 */
  hub: Map<string, number>
  trees: Tree[]
  branches: Branch[]
  lessons: Lesson[]
  byId: Map<string, Lesson>
  /** direct dependents: id -> ids that list it in `requires` */
  unlocks: Map<string, string[]>
}

export type Status = 'todo' | 'in_progress' | 'done'

/** status shown on the graph: learner-set status plus derived availability */
export type NodeState = 'done' | 'in_progress' | 'available' | 'locked'
