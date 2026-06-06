import { Nav } from "@/components/Nav"
import { Hero } from "@/components/Hero"
import { ProblemSection } from "@/components/ProblemSection"
import { ApproachSection } from "@/components/ApproachSection"
import { QuantumAdvantageSection } from "@/components/QuantumAdvantageSection"
import { NetworkSection } from "@/components/NetworkSection"
import { ClaimsSection } from "@/components/ClaimsSection"
import { Footer } from "@/components/Footer"

export function LandingPage() {
  return (
    <main
      id="top"
      className="min-h-screen scroll-smooth bg-background text-foreground antialiased"
    >
      <Nav />
      <Hero />
      <ProblemSection />
      <ApproachSection />
      <QuantumAdvantageSection />
      <NetworkSection />
      <ClaimsSection />
      <Footer />
    </main>
  )
}
