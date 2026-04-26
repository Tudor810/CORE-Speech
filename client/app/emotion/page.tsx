import EmotionSelector from "@/components/EmotionSelector";

export default function EmotionPage() {
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold">Mood</h1>
        <p className="text-slate-400 text-sm">
          Choose how your voice should feel — calm, friendly, urgent, and more.
        </p>
      </header>

      <section className="card">
        <EmotionSelector />
      </section>
    </div>
  );
}
