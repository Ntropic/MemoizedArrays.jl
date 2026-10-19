using Documenter, MemoizedArrays

makedocs(
    sitename="MemoizedArrays.jl", modules=[MemoizedArrays], checkdocs=:exports, remotes=nothing,
    format=Documenter.HTML(prettyurls=false, edit_link=nothing, repolink=nothing),
    pages=["Overview" => "index.md", "Examples" => "examples.md", "API reference" => [
        "Overview" => "api.md", "Caches" => "api/caches.md", "Maps" => "api/maps.md", "Array interface" => "api/interface.md",
    ], "Benchmarks" => "benchmarks.md"],
)
