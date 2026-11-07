using Documenter, MemoizedArrays

makedocs(
    sitename="MemoizedArrays.jl", modules=[MemoizedArrays], checkdocs=:exports, remotes=nothing,
    format=Documenter.HTML(prettyurls=false, edit_link=nothing, repolink=nothing,
        assets=["assets/logo.css"]),
    pages=["Overview" => "index.md", "Examples" => "examples.md", "API reference" => [
        "Overview" => "api.md", "Caches" => "api/caches.md", "Maps" => "api/maps.md", "Array interface" => "api/interface.md",
    ], "Benchmarks" => "benchmarks.md"],
)

# Documenter reserves `logo.svg` for sidebar branding; keep package wordmarks and
# standalone icons separately so the sidebar can use the latter.
function inject_sidebar_icons!(build)
    marker = "<nav class=\"docs-sidebar\">"
    for (directory, _, files) in walkdir(build), file in files
        endswith(file, ".html") || continue
        page = joinpath(directory, file)
        html = read(page, String)
        occursin(marker, html) || continue
        occursin("class=\"docs-logo\"", html) && continue
        root = replace(relpath(joinpath(build, "index.html"), directory), '\\' => '/')
        light = replace(relpath(joinpath(build, "assets", "icon-light.svg"), directory), '\\' => '/')
        dark = replace(relpath(joinpath(build, "assets", "icon-dark.svg"), directory), '\\' => '/')
        icon = "<a class=\"docs-logo\" href=\"$root\"><img class=\"docs-light-only\" src=\"$light\" alt=\"MemoizedArrays.jl icon\"/><img class=\"docs-dark-only\" src=\"$dark\" alt=\"MemoizedArrays.jl icon\"/></a>"
        write(page, replace(html, marker => marker * icon; count=1))
    end
end

inject_sidebar_icons!(joinpath(@__DIR__, "build"))
