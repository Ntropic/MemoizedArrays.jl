using Documenter, JITArrays

makedocs(
    sitename="JITArrays.jl", modules=[JITArrays], checkdocs=:exports, remotes=nothing,
    format=Documenter.HTML(prettyurls=false, edit_link=nothing, repolink=nothing),
    pages=["Overview" => "index.md", "API reference" => [
        "Overview" => "api.md", "Array interface" => "api/interface.md", "Caches" => "api/caches.md", "Storage" => "api/storage.md",
    ], "Benchmarks" => "benchmarks.md"],
)
