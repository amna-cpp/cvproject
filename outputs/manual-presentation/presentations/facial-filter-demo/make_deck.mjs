import fs from "node:fs/promises";
import path from "node:path";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "/Users/amnahameed/.codex/plugins/cache/openai-primary-runtime/presentations/26.521.10419/skills/presentations/scripts/artifact_tool_utils.mjs";

const ROOT = "/Users/amnahameed/Downloads/cv/facial_filter_project";
const WORKSPACE = path.join(ROOT, "outputs/manual-presentation/presentations/facial-filter-demo");
const OUTPUT_DIR = path.join(ROOT, "presentation");
const PREVIEW_DIR = path.join(WORKSPACE, "preview");
const LAYOUT_DIR = path.join(WORKSPACE, "layout");
const CONTACT_SHEET = path.join(WORKSPACE, "contact_sheet.png");
const FINAL_PPTX = path.join(OUTPUT_DIR, "Facial_Filter_Impact_Presentation.pptx");

const PLOTS = path.join(ROOT, "implementation/results/plots");

const SLIDE = { width: 1280, height: 720 };
const C = {
  bg: "#F7F8FA",
  ink: "#111111",
  muted: "#4B5563",
  line: "#D9DEE8",
  panel: "#FFFFFF",
  blue: "#2563EB",
  green: "#16A34A",
  orange: "#EA580C",
  red: "#DC2626",
};

await ensureArtifactToolWorkspace(WORKSPACE);
const artifact = await importArtifactTool(WORKSPACE);
const { Presentation, PresentationFile } = artifact;
const presentation = Presentation.create({ slideSize: SLIDE });

function ctxFor(n) {
  return createSlideContext(artifact, {
    slideSize: SLIDE,
    slideNumber: n,
    outputDir: OUTPUT_DIR,
    assetDir: path.join(WORKSPACE, "assets"),
    workspaceDir: WORKSPACE,
  });
}

function addSlide() {
  const slide = presentation.slides.add();
  const ctx = ctxFor(presentation.slides.count);
  ctx.addShape(slide, { x: 0, y: 0, width: SLIDE.width, height: SLIDE.height, fill: C.bg });
  return { slide, ctx };
}

function text(ctx, slide, s, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    text: s,
    x,
    y,
    width: w,
    height: h,
    fontSize: opts.size ?? 28,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    typeface: opts.face ?? "Aptos",
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    fill: opts.fill ?? "#00000000",
    line: { style: "solid", fill: "#00000000", width: 0 },
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

function title(ctx, slide, s, sub = "") {
  text(ctx, slide, s, 70, 46, 900, 62, { size: 34, bold: true });
  if (sub) text(ctx, slide, sub, 72, 106, 880, 35, { size: 16, color: C.muted });
  ctx.addShape(slide, { x: 70, y: 150, width: 1140, height: 2, fill: C.line });
}

function footer(ctx, slide, n) {
  text(ctx, slide, "Facial Filter Impact on Face Recognition", 70, 674, 520, 20, { size: 11, color: C.muted });
  text(ctx, slide, String(n).padStart(2, "0"), 1160, 674, 50, 20, { size: 11, color: C.muted, align: "right" });
}

function bullet(ctx, slide, s, x, y, w, opts = {}) {
  const color = opts.color ?? C.ink;
  ctx.addShape(slide, { x, y: y + 10, width: 9, height: 9, fill: opts.dot ?? C.blue, geometry: "ellipse" });
  text(ctx, slide, s, x + 24, y, w - 24, opts.h ?? 34, { size: opts.size ?? 21, color });
}

function panel(ctx, slide, x, y, w, h, opts = {}) {
  ctx.addShape(slide, {
    x,
    y,
    width: w,
    height: h,
    fill: opts.fill ?? C.panel,
    line: { style: "solid", fill: opts.line ?? C.line, width: 1 },
    geometry: "roundRect",
  });
}

function stat(ctx, slide, label, value, x, y, w, color = C.blue) {
  panel(ctx, slide, x, y, w, 116);
  text(ctx, slide, value, x + 22, y + 20, w - 44, 42, { size: 32, bold: true, color });
  text(ctx, slide, label, x + 22, y + 67, w - 44, 32, { size: 14, color: C.muted });
}

async function image(ctx, slide, rel, x, y, w, h, fit = "contain") {
  await ctx.addImage(slide, {
    path: path.join(PLOTS, rel),
    x,
    y,
    width: w,
    height: h,
    fit,
    alt: rel,
  });
}

function arrow(ctx, slide, x1, y1, x2, y2, color = C.line) {
  ctx.addShape(slide, { x: x1, y: y1, width: x2 - x1, height: 2, fill: color });
  ctx.addShape(slide, { x: x2 - 10, y: y2 - 6, width: 12, height: 12, fill: color, geometry: "triangle" });
}

// 1
{
  const { slide, ctx } = addSlide();
  text(ctx, slide, "Facial Filter Impact", 76, 140, 770, 70, { size: 54, bold: true });
  text(ctx, slide, "on Face Recognition", 78, 208, 760, 62, { size: 44, bold: true, color: C.blue });
  text(ctx, slide, "End-to-end computer vision system", 82, 300, 620, 34, { size: 22, color: C.muted });
  text(ctx, slide, "Dataset → filters → embeddings → evaluation → demo", 82, 344, 720, 32, { size: 20 });
  stat(ctx, slide, "evaluation pairs", "600", 860, 150, 260, C.blue);
  stat(ctx, slide, "filter types", "6", 860, 292, 260, C.orange);
  stat(ctx, slide, "live Gradio demo", "Ready", 860, 434, 260, C.green);
  text(ctx, slide, "Computer Vision | Spring 2026", 82, 610, 560, 24, { size: 16, color: C.muted });
  footer(ctx, slide, 1);
}

// 2
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Problem Overview", "Face filters can change what the model sees");
  bullet(ctx, slide, "Filters change color, skin texture, eyes, and face shape", 110, 230, 760);
  bullet(ctx, slide, "Humans still recognize the person", 110, 300, 760);
  bullet(ctx, slide, "Face models may shift the embedding", 110, 370, 760);
  bullet(ctx, slide, "This can increase false non-matches", 110, 440, 760, { dot: C.red });
  panel(ctx, slide, 880, 230, 250, 230);
  text(ctx, slide, "Main Question", 910, 260, 190, 30, { size: 18, bold: true });
  text(ctx, slide, "How much do filters hurt face verification?", 910, 310, 190, 92, { size: 25, bold: true, color: C.blue });
  footer(ctx, slide, 2);
}

// 3
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Research Basis", "We reimplemented the paper's evaluation idea");
  panel(ctx, slide, 90, 210, 330, 270);
  text(ctx, slide, "Paper", 120, 240, 260, 34, { size: 22, bold: true });
  bullet(ctx, slide, "Filter impact study", 125, 300, 240, { size: 20, dot: C.blue });
  bullet(ctx, slide, "FNMR / FMR metrics", 125, 350, 240, { size: 20, dot: C.blue });
  bullet(ctx, slide, "DET curves", 125, 400, 240, { size: 20, dot: C.blue });
  panel(ctx, slide, 475, 210, 330, 270);
  text(ctx, slide, "Our System", 505, 240, 260, 34, { size: 22, bold: true });
  bullet(ctx, slide, "LFW dataset", 510, 300, 240, { size: 20, dot: C.green });
  bullet(ctx, slide, "6 simulated filters", 510, 350, 240, { size: 20, dot: C.green });
  bullet(ctx, slide, "Live app demo", 510, 400, 240, { size: 20, dot: C.green });
  panel(ctx, slide, 860, 210, 330, 270);
  text(ctx, slide, "Added Work", 890, 240, 260, 34, { size: 22, bold: true });
  bullet(ctx, slide, "Improved model path", 895, 300, 250, { size: 20, dot: C.orange });
  bullet(ctx, slide, "Linear corrector", 895, 350, 250, { size: 20, dot: C.orange });
  bullet(ctx, slide, "Dashboard results", 895, 400, 250, { size: 20, dot: C.orange });
  footer(ctx, slide, 3);
}

// 4
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Methodology and Architecture", "One full pipeline from data to demo");
  const steps = [
    ["LFW data", "pairs"],
    ["Filters", "6 types"],
    ["Embeddings", "512-D"],
    ["Evaluate", "FNMR / DET"],
    ["Mitigate", "correction"],
    ["Demo", "Gradio"],
  ];
  let x = 70;
  for (let i = 0; i < steps.length; i++) {
    panel(ctx, slide, x, 255, 170, 130, { fill: "#FFFFFF" });
    text(ctx, slide, steps[i][0], x + 18, 285, 134, 28, { size: 22, bold: true, align: "center" });
    text(ctx, slide, steps[i][1], x + 18, 326, 134, 22, { size: 16, color: C.muted, align: "center" });
    if (i < steps.length - 1) arrow(ctx, slide, x + 176, 320, x + 210, 320, C.blue);
    x += 200;
  }
  text(ctx, slide, "Scripts: 01 setup → 02 filters → 04 embeddings → 05 evaluation → 06 mitigation → app", 120, 465, 1040, 36, { size: 22, align: "center" });
  footer(ctx, slide, 4);
}

// 5
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Dataset and Filters", "Public data + generated filter variants");
  stat(ctx, slide, "original LFW images", "13,233", 90, 215, 245, C.blue);
  stat(ctx, slide, "identities", "5,749", 365, 215, 245, C.blue);
  stat(ctx, slide, "filtered images", "4,842", 640, 215, 245, C.orange);
  stat(ctx, slide, "test pairs", "600", 915, 215, 245, C.green);
  text(ctx, slide, "Filter set", 110, 405, 190, 34, { size: 24, bold: true });
  bullet(ctx, slide, "blur", 130, 465, 250, { size: 20 });
  bullet(ctx, slide, "brightness", 330, 465, 250, { size: 20 });
  bullet(ctx, slide, "skin_smooth", 570, 465, 250, { size: 20 });
  bullet(ctx, slide, "eye_enlarge", 830, 465, 250, { size: 20 });
  bullet(ctx, slide, "face_slim", 330, 525, 250, { size: 20, dot: C.orange });
  bullet(ctx, slide, "color_tone", 570, 525, 250, { size: 20, dot: C.orange });
  footer(ctx, slide, 5);
}

// 6
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Live Demo Plan", "What I will show in the app");
  panel(ctx, slide, 130, 215, 290, 290);
  text(ctx, slide, "1", 155, 235, 50, 50, { size: 34, bold: true, color: C.blue });
  text(ctx, slide, "Upload two faces", 210, 248, 170, 32, { size: 24, bold: true });
  text(ctx, slide, "same or different person", 210, 300, 170, 55, { size: 18, color: C.muted });
  panel(ctx, slide, 495, 215, 290, 290);
  text(ctx, slide, "2", 520, 235, 50, 50, { size: 34, bold: true, color: C.blue });
  text(ctx, slide, "Choose settings", 575, 248, 170, 32, { size: 24, bold: true });
  text(ctx, slide, "model, mitigation, filter", 575, 300, 170, 55, { size: 18, color: C.muted });
  panel(ctx, slide, 860, 215, 290, 290);
  text(ctx, slide, "3", 885, 235, 50, 50, { size: 34, bold: true, color: C.blue });
  text(ctx, slide, "Verify identity", 940, 248, 170, 32, { size: 24, bold: true });
  text(ctx, slide, "similarity + verdict", 940, 300, 170, 55, { size: 18, color: C.muted });
  text(ctx, slide, "Demo URL: http://127.0.0.1:7860", 305, 570, 670, 30, { size: 24, bold: true, align: "center" });
  footer(ctx, slide, 6);
}

// 7
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Baseline Results", "ArcFace under simulated filters");
  await image(ctx, slide, "accuracy_per_filter.png", 80, 190, 700, 390, "contain");
  panel(ctx, slide, 840, 210, 300, 300);
  text(ctx, slide, "Key numbers", 870, 240, 230, 30, { size: 22, bold: true });
  bullet(ctx, slide, "Average accuracy: 58.81%", 875, 305, 220, { size: 20, dot: C.blue });
  bullet(ctx, slide, "Average FNMR@1%: 94.78%", 875, 365, 220, { size: 20, dot: C.red });
  bullet(ctx, slide, "High miss rate", 875, 425, 220, { size: 20, dot: C.red });
  footer(ctx, slide, 7);
}

// 8
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Improved Model Comparison", "Better backbone path improves accuracy");
  await image(ctx, slide, "comparison/accuracy_4way_comparison.png", 85, 178, 760, 430, "contain");
  panel(ctx, slide, 895, 230, 250, 230);
  text(ctx, slide, "Result", 925, 260, 190, 30, { size: 22, bold: true });
  text(ctx, slide, "58.81% → 77.86%", 925, 315, 190, 50, { size: 27, bold: true, color: C.green });
  text(ctx, slide, "average filtered accuracy", 925, 380, 190, 45, { size: 17, color: C.muted });
  footer(ctx, slide, 8);
}

// 9
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Error Rate and Mitigation", "Lower FNMR is better");
  await image(ctx, slide, "comparison/fnmr_improvement.png", 90, 185, 710, 380, "contain");
  await image(ctx, slide, "comparison/summary_table.png", 835, 210, 340, 300, "contain");
  text(ctx, slide, "Average FNMR reduction", 870, 545, 280, 28, { size: 20, bold: true, align: "center" });
  text(ctx, slide, "40.2% vs ArcFace baseline", 870, 580, 280, 30, { size: 22, bold: true, color: C.green, align: "center" });
  footer(ctx, slide, 9);
}

// 10
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Key Insights", "What we learned");
  panel(ctx, slide, 110, 220, 305, 245);
  text(ctx, slide, "1", 135, 240, 40, 40, { size: 32, bold: true, color: C.blue });
  text(ctx, slide, "Filters change embeddings", 185, 248, 190, 60, { size: 25, bold: true });
  text(ctx, slide, "even when the face still looks clear", 185, 335, 190, 55, { size: 17, color: C.muted });
  panel(ctx, slide, 485, 220, 305, 245);
  text(ctx, slide, "2", 510, 240, 40, 40, { size: 32, bold: true, color: C.blue });
  text(ctx, slide, "Model choice matters", 560, 248, 190, 60, { size: 25, bold: true });
  text(ctx, slide, "the improved path was stronger here", 560, 335, 190, 55, { size: 17, color: C.muted });
  panel(ctx, slide, 860, 220, 305, 245);
  text(ctx, slide, "3", 885, 240, 40, 40, { size: 32, bold: true, color: C.blue });
  text(ctx, slide, "Correction is useful", 935, 248, 190, 60, { size: 25, bold: true });
  text(ctx, slide, "but filter-specific and not perfect", 935, 335, 190, 55, { size: 17, color: C.muted });
  footer(ctx, slide, 10);
}

// 11
{
  const { slide, ctx } = addSlide();
  title(ctx, slide, "Limitations and Future Work", "Honest scope of this prototype");
  text(ctx, slide, "Limitations", 120, 215, 400, 34, { size: 26, bold: true });
  bullet(ctx, slide, "LFW, not FRGC", 130, 280, 470, { size: 22, dot: C.red });
  bullet(ctx, slide, "Programmatic filters, not real app filters", 130, 340, 470, { size: 22, dot: C.red });
  bullet(ctx, slide, "600 pair subset", 130, 400, 470, { size: 22, dot: C.red });
  bullet(ctx, slide, "No fairness testing yet", 130, 460, 470, { size: 22, dot: C.red });
  text(ctx, slide, "Next Steps", 720, 215, 400, 34, { size: 26, bold: true });
  bullet(ctx, slide, "Use FRGC dataset", 730, 280, 430, { size: 22, dot: C.green });
  bullet(ctx, slide, "Apply real AR filters", 730, 340, 430, { size: 22, dot: C.green });
  bullet(ctx, slide, "Train one filter-agnostic corrector", 730, 400, 430, { size: 22, dot: C.green });
  bullet(ctx, slide, "Study gender and skin tone effects", 730, 460, 430, { size: 22, dot: C.green });
  footer(ctx, slide, 11);
}

// 12
{
  const { slide, ctx } = addSlide();
  text(ctx, slide, "Thank You", 80, 170, 720, 70, { size: 60, bold: true });
  text(ctx, slide, "Questions?", 84, 250, 500, 55, { size: 42, bold: true, color: C.blue });
  panel(ctx, slide, 760, 170, 350, 310);
  text(ctx, slide, "Ready for demo", 800, 215, 260, 36, { size: 28, bold: true, color: C.green, align: "center" });
  text(ctx, slide, "Upload images", 800, 295, 260, 28, { size: 24, align: "center" });
  text(ctx, slide, "Choose filter", 800, 330, 260, 28, { size: 24, align: "center" });
  text(ctx, slide, "Compare result", 800, 365, 260, 28, { size: 24, align: "center" });
  text(ctx, slide, "http://127.0.0.1:7860", 80, 570, 520, 30, { size: 24, bold: true });
  footer(ctx, slide, 12);
}

await fs.mkdir(PREVIEW_DIR, { recursive: true });
await fs.mkdir(LAYOUT_DIR, { recursive: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });

const previewPaths = [];
for (let i = 0; i < presentation.slides.count; i++) {
  const slide = presentation.slides.getItem(i);
  const previewPath = path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await saveBlobToFile(preview, previewPath);
  previewPaths.push(previewPath);
  const layout = await presentation.export({ slide, format: "layout" });
  await fs.writeFile(path.join(LAYOUT_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text(), "utf8");
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL_PPTX);

await fs.writeFile(
  path.join(WORKSPACE, "profile-plan.txt"),
  [
    "task mode: create",
    "primary deck-profile: engineering-platform",
    "proof objects: real metrics JSON, accuracy plots, FNMR plot, summary table",
    "QA gates: short text, live demo flow, readable charts, final PPTX export",
  ].join("\\n") + "\\n",
  "utf8",
);

console.log(JSON.stringify({ output: FINAL_PPTX, slideCount: presentation.slides.count, previewPaths }, null, 2));
