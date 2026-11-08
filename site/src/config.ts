// Single source of truth for site settings. Curriculum content lives in ../lessons/.

export const CONFIG = {
  repoUrl: 'https://github.com/woolimi/robotics-roadmap',
  repoBranch: 'main',

  // link previews on LinkedIn, KakaoTalk and Slack (Open Graph). Crawlers need absolute URLs, so they point at the
  // published example site; vite.config.ts writes these into index.html
  share: {
    siteUrl: 'https://woolimi.github.io/robotics-roadmap/',
    title: 'Robotics Roadmap',
    description: '로보틱스 개발자가 되기 위한 오픈소스 독학 커리큘럼. 강의 56개를 선수 관계 지도로 따라가며 읽고 면접형 퀴즈로 확인합니다.',
    // site/public/og.png. Link previews are cached by the image's URL, so bump `v` whenever the picture changes
    image: 'og.png?v=2',
    imageWidth: 1200,
    imageHeight: 630,
    imageAlt: 'Robotics Roadmap 커리큘럼 지도. 가운데 공통 기초 동심원에서 Robot Learning, Manipulation, Mobile Autonomy 트리가 뻗는다',
    locale: 'ko_KR',
  },

  storage: {
    introSeenKey: 'pafs.intro-seen.v1',
    panelWidthKey: 'pafs.panel-width.v1',
  },

  lesson: {
    sectionIdPrefix: 'sec-', // a lesson heading's id is this plus its GitHub slug (lesson-render.ts)
    videoEmbed: 'https://www.youtube-nocookie.com/embed/', // + the frontmatter video's id; the player sets no cookies until it plays
  },

  // lesson panel on wide screens; its left edge drags to resize
  panel: {
    defaultWidth: 600,
    minWidth: 380,
    edgeGutter: 48, // the panel may grow over the sidebar and graph up to this strip at the window's left edge
    keyStep: 32, // arrow keys on the resize handle
    prefetchDelay: 1500, // ms after the map shows to fetch the panel's code where the browser has no requestIdleCallback (Safari)
  },

  // multiple-choice quiz that closes each lesson. The dev server grades a submission and appends it to the lesson's
  // progress file; a lesson is done once one attempt gets every answer right.
  quiz: {
    bankFile: 'quiz.yaml', // lessons/<id>/quiz.yaml; the globs in curriculum/load.ts must use the same names (globs take literals only)
    progressDir: 'progress', // <repo>/progress/<id>.json, kept apart from lessons/ so a fork commits only its own records; read by lesson-data.ts
    endpoint: '/__quiz', // POST /__quiz/<id>, served by the dev server only
    choiceCount: 4,
    reviewFocusGap: 16, // px of the review left above the explanation a learner comes back to from the lecture
  },

  // when the dev server starts it checks the original repository (repoUrl) for commits the fork does not have yet, and
  // the page offers to merge them
  update: {
    remote: 'upstream', // the git remote the original is fetched into; added to the fork if it is missing
    endpoint: '/__update', // GET: the fork's state; POST: merge the original. Served by the dev server only
    event: 'roadmap:update', // Vite WebSocket event that carries every change of state to the page
    fetchTimeoutMs: 20_000,
    gitTimeoutMs: 10_000,
    // Vite restarts by itself when its config or server code changes, but new or upgraded dependencies need npm install
    packageFile: 'site/package.json', // its dependencies must be in the node_modules next to it, at the versions in lockFile
    lockFile: 'site/package-lock.json',
  },

  // layered map from ELK: prerequisites above, the lessons they open below
  layout: {
    nodeRadius: { small: 13, medium: 16, large: 19 },
    hoursForSize: { medium: 12, large: 25 }, // hours >= threshold
    labelFontSize: 13,
    labelLineHeight: 15,
    labelCharWidth: 7.9, // JetBrains Mono advance at labelFontSize
    labelWrapAt: 12, // titles longer than this wrap onto two lines
    labelOffset: 7, // gap between node circle and its label
    boxPad: 10, // breathing room on each side of a node+label box
    startTagHeight: 18, // space reserved above the starting lesson for its START tag
    nodeGap: 64, // px between neighbouring lessons in a layer
    layerGap: 96, // px between layers
    edgeNodeGap: 14, // px between an edge and a lesson it passes
    edgeGap: 20, // px between parallel edges and between the waypoints of long edges
    trunkStraightness: 10, // ELK priority for keeping core-chain edges straight
    curveSamples: 12, // points per Bézier segment when an edge is sampled
    orderRestarts: 100, // random starts of the crossing minimisation; 100 settles on the same crossing count for any seed
    orderSweeps: 8, // barycentre sweeps per start
    orderSeed: 1,
    hubRing: 60, // px from the start to the first ring of hub lessons around it, and between further rings
    hubExit: 0.8, // how far to the side of a tree's axis an edge may leave the outermost ring, as a share of its radius
    hubPullAgreement: 0.4, // a hub lesson faces its trees only when their directions agree this much (1: all the same way)
    hubTurnStep: 2, // degrees two touching hub lessons on one ring turn apart per round
    treeGap: 24, // least px between the labels of lessons in different trees
    pushStep: 8, // px a tree moves away from the centre per round while its labels touch another tree's
    pushRounds: 200, // rounds of pushing before the layout gives up
    edgeBend: 0.2, // how far an edge between two trees reaches along each tree's direction, as a share of its length
  },

  sector: {
    charWidth: 66, // px each character of the name may take, letter spacing included (styles.css .hg-sector)
    lineHeight: 96,
    artScale: 1.15, // the tree's picture is this many times the least square, centred on it, that holds its lessons
  },

  // the WebGL layer under the map (components/MapScene.tsx): clouds in each tree's colour
  scene: {
    fps: 30, // the clouds drift slowly, so they are drawn at most this often
    maxPixelRatio: 2, // caps the canvas resolution on high-density screens
    nebulaScale: 700, // px of the layout per noise cell of the clouds
    nebulaStrength: 0.3,
    // trees whose clouds are thinned near the hub, by this share at the rim easing to none at three times its radius, so
    // the clouds stay clear of the hub's lessons; amber reads much brighter than the other colours there
    innerCut: { learning: 0.4, manipulation: 0.4, mobile: 0.6 } as Record<string, number>,
    treeDensity: { mobile: 0.7 } as Record<string, number>, // a tree's clouds, as a share of the others' brightness
    nebulaSpread: 1.7, // the clouds fade out by this many times the distance from the centre to the map's farthest point
    // stars behind the clouds, scattered over the same square; most are small, a few large
    stars: { count: 1400, seed: 7, minSize: 1.6, maxSize: 4.4, color: '#dbe7ff', brightness: 1.5, blinkShare: 0.06 },
    // a shooting star every `every` seconds (a random wait between the two), `duration` seconds long, crossing `travel`
    // screen px at `angle`±`angleSpread` degrees below the horizontal, left or right
    meteors: {
      firstAfter: 3,
      every: [6, 14],
      duration: 0.9,
      travel: 520,
      length: 180, // screen px of the streak
      width: 3,
      angle: 30,
      angleSpread: 12,
      color: '#e6f4ff',
      brightness: 2,
    },
    accent: '#22d3ee', // the hovered hub's light, in the site's accent colour (styles.css --accent)
    focusGain: 1.2, // how much brighter the hovered area's clouds get (1.2: 2.2 times)
    hubGlow: 0.07, // how bright the accent cloud filling the hovered hub is, where its clouds are thinnest
    focusEase: 6, // how fast an area's light follows the pointer, per second
  },

  zoom: {
    min: 0.25,
    max: 3,
    labelMinPx: 10, // zoomed out, labels grow to keep at least this on-screen size ...
    labelMaxGrow: 1.5, // ... but never past this many times their size, where neighbouring labels would touch
    initialMinScale: 0.85, // first view never zooms out further than this, so labels stay readable
    fitPadding: 40,
  },
} as const

export type Config = typeof CONFIG
