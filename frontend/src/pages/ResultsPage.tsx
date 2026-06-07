export function ResultsPage() {
  // BASE_URL is "/" in dev and "/<repo>/" on GitHub Pages, so the iframe
  // resolves under the project subpath instead of the domain root.
  const proto = `${import.meta.env.BASE_URL}proto/index.html`
  return (
    <main className="h-screen w-screen overflow-hidden bg-[#0b0f1a]">
      <iframe
        src={proto}
        title="Failure network"
        className="block h-full w-full border-0"
      />
    </main>
  )
}
