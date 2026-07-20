import { useTheme } from "../lib/useTheme";
import { Moon, Sun } from "./icons";

export default function ThemeToggle() {
  const [theme, toggle] = useTheme();
  return (
    <button
      onClick={toggle}
      className="btn-ghost !p-2.5"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title="Toggle theme"
    >
      {theme === "dark" ? <Sun width={18} height={18} /> : <Moon width={18} height={18} />}
    </button>
  );
}
