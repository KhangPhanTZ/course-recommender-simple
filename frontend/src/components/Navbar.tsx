import { useState } from "react";
import { NavLink, Link } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";
import { Close, Github, Menu, Sparkles } from "./icons";

const links = [
  { to: "/search", label: "Search" },
  { to: "/chat", label: "Chat" },
  { to: "/explore", label: "Explore" },
  { to: "/map", label: "Map" },
  { to: "/architecture", label: "Architecture" },
  { to: "/about", label: "About" },
];

const REPO = "https://github.com/KhangPhanTZ/course-recommender-simple";

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-[rgb(var(--border))] bg-[rgb(var(--bg))]/80 backdrop-blur-lg">
      <nav className="container-page flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-600 text-white shadow-sm">
            <Sparkles width={18} height={18} />
          </span>
          <span className="text-[rgb(var(--text))]">
            Path<span className="text-brand-600 dark:text-brand-400">finder</span>
          </span>
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-[rgb(var(--surface-2))] text-brand-700 dark:text-brand-300"
                    : "text-body hover:text-[rgb(var(--text))]"
                }`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <a href={REPO} target="_blank" rel="noreferrer" className="btn-ghost !p-2.5" aria-label="GitHub repository">
            <Github width={18} height={18} />
          </a>
          <ThemeToggle />
          <Link to="/search" className="btn-primary hidden sm:inline-flex">
            Try it
          </Link>
          <button
            className="btn-ghost !p-2.5 md:hidden"
            onClick={() => setOpen((o) => !o)}
            aria-label="Toggle menu"
          >
            {open ? <Close width={18} height={18} /> : <Menu width={18} height={18} />}
          </button>
        </div>
      </nav>

      {open && (
        <div className="border-t border-[rgb(var(--border))] md:hidden">
          <div className="container-page flex flex-col gap-1 py-3">
            {links.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2.5 text-sm font-medium ${
                    isActive ? "bg-[rgb(var(--surface-2))] text-brand-700 dark:text-brand-300" : "text-body"
                  }`
                }
              >
                {l.label}
              </NavLink>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
