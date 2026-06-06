import { Atom } from "lucide-react"
import { Link } from "react-router-dom"
import { buttonVariants } from "@/components/ui/button"

const links = [
  { href: "#problem", label: "Problem" },
  { href: "#approach", label: "Approach" },
  { href: "#quantum", label: "Quantum" },
  { href: "#network", label: "Network" },
  { href: "#claims", label: "Scope" },
]

export function Nav() {
  const go = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault()
    document
      .querySelector(href)
      ?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-border/60 bg-background/70 backdrop-blur-md">
      <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault()
            window.scrollTo({ top: 0, behavior: "smooth" })
          }}
          className="flex items-center gap-2 font-semibold tracking-tight"
        >
          <Atom className="size-5 text-indigo-400" />
          <span className="hidden sm:inline">Quantum Stress</span>
        </a>
        <ul className="flex items-center gap-1 text-sm">
          {links.map((l) => (
            <li key={l.href} className="hidden sm:block">
              <a
                href={l.href}
                onClick={(e) => go(e, l.href)}
                className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                {l.label}
              </a>
            </li>
          ))}
          <li className="ml-2">
            <Link to="/results" className={buttonVariants({ size: "sm" })}>
              Results
            </Link>
          </li>
        </ul>
      </nav>
    </header>
  )
}
