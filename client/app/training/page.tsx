import TrainingStudio from "@/components/TrainingStudio";

export default function TrainingPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Teach signs</h1>
        <p className="text-slate-400 text-sm">
          Show a sign in front of the camera a few times to teach it to your glasses.
        </p>
      </header>

      <TrainingStudio />
    </div>
  );
}
