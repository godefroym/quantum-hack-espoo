import { Route, Routes } from "react-router-dom"
import { LandingPage } from "@/pages/LandingPage"
import { ResultsPage } from "@/pages/ResultsPage"

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/results" element={<ResultsPage />} />
    </Routes>
  )
}

export default App
