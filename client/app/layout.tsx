import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AR Glasses Companion",
  description:
    "Configuration dashboard for the Deaf/Mute AR Glasses sign-speech bridge.",
};

const NAV = [
  { href: "/", label: "Home" },
  { href: "/live", label: "Live" },
  { href: "/voice", label: "Voice" },
  { href: "/emotion", label: "Mood" },
  { href: "/training", label: "Teach signs" },
] as const;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-border bg-panel/60 backdrop-blur sticky top-0 z-10">
            <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-6">
              <Link href="/" className="font-semibold tracking-tight">
                <span className="text-accent">AR</span> Companion
              </Link>
              <nav className="flex gap-4 text-sm text-slate-300">
                {NAV.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="hover:text-white"
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>
            </div>
          </header>
          <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-8">
            {children}
          </main>
          <footer className="border-t border-border text-xs text-slate-500 py-4 text-center">
            AR Glasses Companion
          </footer>
        </div>
      </body>
    </html>
  );
}
