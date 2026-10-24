pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "GemmaVoiceChat"

include(":app")
include(":sherpa")
include(":lib")

val llamaLib = file("../third_party/llama.cpp/examples/llama.android/lib")
if (!llamaLib.isDirectory) {
    error(
        "Missing llama.cpp lib module. Run ../scripts/bootstrap.sh from the repo root first."
    )
}
project(":lib").projectDir = llamaLib
