package com.gemmaproj.voicechat.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.gemmaproj.voicechat.GemmaVoiceApp
import com.gemmaproj.voicechat.data.ModelPaths
import com.gemmaproj.voicechat.domain.ChatMessage
import com.gemmaproj.voicechat.domain.ChatMode
import com.gemmaproj.voicechat.domain.GemmaVariant
import com.gemmaproj.voicechat.domain.SttEngine
import com.gemmaproj.voicechat.domain.UserPreferences
import com.gemmaproj.voicechat.domain.VoiceStack
import com.gemmaproj.voicechat.engine.ApproachPilot
import com.gemmaproj.voicechat.engine.AttractSound
import com.gemmaproj.voicechat.engine.DogSpeech
import com.gemmaproj.voicechat.engine.EmbeddingLocator
import com.gemmaproj.voicechat.engine.PetoiMover
import com.gemmaproj.voicechat.engine.PlayWatcher
import com.gemmaproj.voicechat.engine.YoloDetector
import com.gemmaproj.voicechat.engine.petoiSkill
import com.google.ai.edge.litertlm.ActivationDataType
import com.gemmaproj.voicechat.engine.LiteRtBench
import com.gemmaproj.voicechat.engine.LiteRtChatEngine
import com.gemmaproj.voicechat.engine.LlamaChatEngine
import com.gemmaproj.voicechat.engine.SceneImages
import com.gemmaproj.voicechat.engine.SpeechToTextEngine
import com.gemmaproj.voicechat.engine.SpokenChunker
import com.gemmaproj.voicechat.engine.TextToSpeechEngine
import com.gemmaproj.voicechat.engine.VoiceRecorder
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class TrackBox(
    val label: String,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
    val isDog: Boolean,
    val role: String = "target",
)

data class ChatUiState(
    val messages: List<ChatMessage> = emptyList(),
    val prefs: UserPreferences = UserPreferences(),
    val status: String = "Loading Gemma…",
    val isReady: Boolean = false,
    val isBusy: Boolean = false,
    val isRecording: Boolean = false,
    val voiceReady: Boolean = false,
    val visionReady: Boolean = false,
    val cameraOpen: Boolean = false,
    val autoScene: Boolean = false,
    val scenePhotoPath: String = "",
    val sceneQuestion: String = "",
    val sceneCaption: String = "",
    val missingModels: List<String> = emptyList(),
    val modelLabel: String = "E2B",
    val trackReady: Boolean = false,
    val trackLive: Boolean = false,
    val trackTarget: String = TRACK_BOTTLE,
    val trackBoxes: List<TrackBox> = emptyList(),
    val trackImageWidth: Int = 0,
    val trackImageHeight: Int = 0,
    val approachLive: Boolean = false,
    /** Normalized x1, y1, x2, y2 of the robot's measured heading, or empty. */
    val headingLine: List<Float> = emptyList(),
    val dogHeard: String = "",
    val dogSaid: String = "",
    val dogDid: String = "",
    val dogTalkNote: String = "Hold Speak to talk to the dog.",
)

internal const val TRACK_PERSON = "a person"
internal const val TRACK_BOTTLE = "a water bottle"

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val paths = ModelPaths.from(application)
    private val llm = LlamaChatEngine(application)
    private val gpu = LiteRtChatEngine(application)
    private val embed = EmbeddingLocator(application)
    private val yolo = YoloDetector()
    private val petoi = PetoiMover()
    private val dog = (application as GemmaVoiceApp).dogLink
    private val play = PlayWatcher()
    private val pilot = ApproachPilot()
    private val pendingFrame = AtomicReference<android.graphics.Bitmap?>(null)
    private var usingGpu = false
    private var chatPausedForTrack = false
    private val stt = SpeechToTextEngine()
    private val tts = TextToSpeechEngine()
    private val recorder = VoiceRecorder()

    private val _uiState = MutableStateFlow(
        ChatUiState(missingModels = paths.missingSummary(GemmaVariant.E2B)),
    )
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null
    private var recordJob: Job? = null
    private var loadJob: Job? = null
    private var locateJob: Job? = null
    private var playJob: Job? = null
    private var moveJob: Job? = null
    private var approachJob: Job? = null
    private var speechError: String? = null

    init {
        val app = getApplication<GemmaVoiceApp>()
        if (app.bleProbe) {
            app.bleProbe = false
            runBleProbe()
        } else if (app.profileTrack) {
            app.profileTrack = false
            val gpu = app.profileGpu
            app.profileGpu = false
            runTrackProfile(gpu)
        } else if (app.litertBench && app.benchVariant == "G3") {
            app.litertBench = false
            app.benchVariant = null
            runGemma3NpuBench()
        } else if (app.litertBench) {
            val variant = if (app.benchVariant == "E4B") GemmaVariant.E4B else GemmaVariant.E2B
            app.litertBench = false
            app.benchVariant = null
            runLitertBench(variant)
        } else if (app.benchVariant == "G3") {
            app.benchVariant = null
            runGemma3LlamaBench()
        } else {
            val requested = app.benchVariant
            var prefs = _uiState.value.prefs
            if (requested == "E2B" || requested == "E4B") {
                prefs = prefs.copy(gemma = GemmaVariant.valueOf(requested))
            }
            if (app.gemmaAudio) {
                prefs = prefs.copy(voiceStack = VoiceStack.NATIVE)
            }
            if (prefs != _uiState.value.prefs) {
                _uiState.value = _uiState.value.copy(prefs = prefs)
            }
            loadModels()
        }
    }

    private fun runLitertBench(variant: GemmaVariant) {
        _uiState.value = _uiState.value.copy(
            isBusy = true,
            isReady = false,
            status = "Loading LiteRT-LM ${variant.name}…",
        )
        viewModelScope.launch {
            try {
                val report = LiteRtBench(getApplication()).run(paths.litertFile(variant))
                Log.i(TAG, "LITERT\n$report")
                java.io.File(getApplication<Application>().filesDir, "bench-litert-${variant.name}.txt")
                    .writeText(report)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = true,
                    status = "LiteRT ${variant.name} done",
                    messages = listOf(
                        ChatMessage(
                            id = UUID.randomUUID().toString(),
                            role = ChatMessage.Role.ASSISTANT,
                            text = report,
                        ),
                    ),
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "litert bench failed", e)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = false,
                    status = "LiteRT failed: ${e.message ?: e.javaClass.simpleName}",
                )
            }
        }
    }

    private fun runGemma3LlamaBench() {
        _uiState.value = _uiState.value.copy(
            isBusy = true,
            isReady = false,
            status = "Loading Gemma 3 1B llama.cpp…",
        )
        viewModelScope.launch {
            try {
                val file = paths.gemma3Gguf()
                val started = System.nanoTime()
                llm.loadFile(file)
                val loadMs = (System.nanoTime() - started) / 1_000_000
                val report = llm.bench(pp = 512, tg = 128, nr = 3)
                val body = "file=${file.name} bytes=${file.length()} load_ms=$loadMs\n$report"
                Log.i(TAG, "BENCH G3\n$body")
                java.io.File(getApplication<Application>().filesDir, "bench-G3.txt").writeText(body)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = true,
                    status = "Gemma 3 llama.cpp done",
                    messages = listOf(ChatMessage(UUID.randomUUID().toString(), ChatMessage.Role.ASSISTANT, body)),
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "gemma3 llama bench failed", e)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = false,
                    status = "Gemma 3 llama.cpp failed: ${e.message ?: e.javaClass.simpleName}",
                )
            }
        }
    }

    private fun runGemma3NpuBench() {
        _uiState.value = _uiState.value.copy(
            isBusy = true,
            isReady = false,
            status = "Loading Gemma 3 1B NPU…",
        )
        viewModelScope.launch {
            try {
                val qnn = stageQnnLibs()
                val report = LiteRtBench(getApplication()).runNpu(paths.gemma3Npu(), qnn)
                Log.i(TAG, "LITERT NPU\n$report")
                java.io.File(getApplication<Application>().filesDir, "bench-litert-G3-npu.txt").writeText(report)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = true,
                    status = "Gemma 3 NPU done",
                    messages = listOf(ChatMessage(UUID.randomUUID().toString(), ChatMessage.Role.ASSISTANT, report)),
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "gemma3 npu bench failed", e)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = false,
                    status = "Gemma 3 NPU failed: ${e.message ?: e.javaClass.simpleName}",
                )
            }
        }
    }

    /**
     * Copies the phone's own Hexagon QNN libraries into app storage.
     * LiteRT's NPU backend dlopens them from this directory, and the DSP
     * skel is found through ADSP_LIBRARY_PATH.
     */
    private fun stageQnnLibs(): java.io.File {
        val dest = getApplication<Application>().filesDir.resolve("qnn")
        dest.mkdirs()
        val sources = listOf(
            "/odm/lib64/libQnnHtp.so",
            "/odm/lib64/libQnnHtpV75Stub.so",
            "/odm/lib64/libQnnSystem.so",
            "/odm/lib64/libQnnHtpPrepare.so",
            "/odm/lib/rfsa/adsp/libQnnHtpV75Skel.so",
        )
        for (sourcePath in sources) {
            val source = java.io.File(sourcePath)
            val out = dest.resolve(source.name)
            if (out.isFile && out.length() == source.length()) continue
            require(source.isFile) { "QNN library not readable: $sourcePath" }
            source.inputStream().use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return dest
    }

    fun refreshModelStatus() {
        val variant = _uiState.value.prefs.gemma
        _uiState.value = _uiState.value.copy(missingModels = paths.missingSummary(variant))
    }

    fun setGemma(variant: GemmaVariant) {
        if (variant == _uiState.value.prefs.gemma && _uiState.value.isReady) return
        _uiState.value = _uiState.value.copy(
            prefs = _uiState.value.prefs.copy(gemma = variant),
            isReady = false,
            missingModels = paths.missingSummary(variant),
            status = "Loading Gemma 4 ${variant.name}…",
        )
        loadModels()
    }

    fun setVoiceStack(stack: VoiceStack) {
        if (stack == _uiState.value.prefs.voiceStack) return
        _uiState.value = _uiState.value.copy(
            prefs = _uiState.value.prefs.copy(voiceStack = stack),
            isReady = false,
        )
        loadModels()
    }

    fun setChatMode(mode: ChatMode) {
        _uiState.value = _uiState.value.copy(prefs = _uiState.value.prefs.copy(chatMode = mode))
    }

    fun setSttEngine(engine: SttEngine) {
        _uiState.value = _uiState.value.copy(prefs = _uiState.value.prefs.copy(sttEngine = engine))
    }

    fun setSpeakReplies(enabled: Boolean) {
        _uiState.value = _uiState.value.copy(prefs = _uiState.value.prefs.copy(speakReplies = enabled))
        if (enabled && _uiState.value.isReady && paths.kokoroReady()) {
            viewModelScope.launch {
                try {
                    tts.ensureLoaded(paths)
                } catch (e: Exception) {
                    Log.e(TAG, "kokoro load failed", e)
                    _uiState.value = _uiState.value.copy(status = "Kokoro: ${e.message}")
                }
            }
        }
    }

    fun setKokoroSpeaker(speakerId: Int) {
        val sid = speakerId.coerceIn(0, 52)
        _uiState.value = _uiState.value.copy(prefs = _uiState.value.prefs.copy(kokoroSpeakerId = sid))
    }

    fun loadModels() {
        loadJob?.cancel()
        locateJob?.cancel()
        stopPlayWatch()
        stopApproach(sendStop = false)
        loadJob = viewModelScope.launch {
            if (embed.isReady) {
                withContext(Dispatchers.IO) { embed.close() }
            }
            chatPausedForTrack = false
            val variant = _uiState.value.prefs.gemma
            _uiState.value = _uiState.value.copy(
                isBusy = true,
                isReady = false,
                visionReady = false,
                cameraOpen = false,
                autoScene = false,
                trackReady = false,
                trackLive = false,
                trackBoxes = emptyList(),
                status = "Loading Gemma 4 ${variant.name}…",
            )
            try {
                val prefs = _uiState.value.prefs
                val listen = prefs.voiceStack == VoiceStack.NATIVE && prefs.gemma == GemmaVariant.E2B
                val audioFile = paths.litertMultimodal()
                val textGpu = paths.litertFile(GemmaVariant.E2B)
                if (listen && !audioFile.isFile) {
                    error("Missing ${audioFile.name}. Gemma audio needs this file.")
                }
                val useGpu = listen || (prefs.gemma == GemmaVariant.E2B && textGpu.isFile)
                val loadStarted = System.nanoTime()
                val modelLabel = if (useGpu) {
                    val file = if (listen) audioFile else textGpu
                    _uiState.value = _uiState.value.copy(
                        status = if (listen) "Loading Gemma 4 E2B audio…" else "Loading Gemma 4 E2B GPU…",
                    )
                    llm.releaseModel()
                    gpu.load(file, prefs.systemPrompt, listen = listen)
                    usingGpu = true
                    if (listen) "E2B audio" else "E2B GPU"
                } else {
                    if (!paths.llmReady(prefs.gemma)) {
                        error("Missing ${paths.llmFile(prefs.gemma).name}")
                    }
                    gpu.close()
                    usingGpu = false
                    llm.load(paths, prefs.gemma)
                    llm.setSystemPrompt(prefs.systemPrompt)
                    prefs.gemma.name
                }
                val loadMs = (System.nanoTime() - loadStarted) / 1_000_000
                val loadLine = "LOAD $modelLabel ${loadMs}ms"
                Log.i(TAG, loadLine)
                java.io.File(getApplication<Application>().filesDir, "hear-log.txt").appendText(loadLine + "\n")
                runBenchIfRequested(prefs, loadMs)
                speechError = null
                val voiceReady = loadSpeechModels()
                _uiState.value = _uiState.value.copy(
                    isReady = true,
                    isBusy = false,
                    voiceReady = voiceReady,
                    modelLabel = modelLabel,
                    status = speechError?.let { "Voice model failed: $it" } ?: "Ready",
                    missingModels = emptyList(),
                )
                runVoiceProbe()
                runAutoAsk()
                runPointPhoto()
            } catch (e: TimeoutCancellationException) {
                Log.e(TAG, "init timed out", e)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = false,
                    status = "Could not load the model: native library timed out",
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "init failed", e)
                val variant = _uiState.value.prefs.gemma
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = false,
                    status = "Could not load the model: ${e.message ?: e.javaClass.simpleName}",
                    missingModels = paths.missingSummary(variant),
                )
            }
        }
    }

    private suspend fun loadSpeechModels(): Boolean {
        val native = _uiState.value.prefs.voiceStack == VoiceStack.NATIVE
        var sttReady = false
        if (native) {
            if (_uiState.value.prefs.speakReplies && paths.kokoroReady()) {
                _uiState.value = _uiState.value.copy(status = "Loading Kokoro…")
                try {
                    tts.ensureLoaded(paths)
                    Log.i(TAG, "Kokoro ready")
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e(TAG, "Kokoro load failed", e)
                    speechError = e.message ?: e.javaClass.simpleName
                }
            }
            return true
        }
        val engine = _uiState.value.prefs.sttEngine
        val sttPresent = when (engine) {
            SttEngine.SENSE_VOICE -> paths.senseVoiceReady()
            SttEngine.WHISPER_TINY_EN -> paths.whisperReady()
        }
        if (sttPresent) {
            _uiState.value = _uiState.value.copy(status = "Loading ${engineLabel(engine)}…")
            try {
                stt.ensureLoaded(paths, engine)
                sttReady = true
                Log.i(TAG, "STT ${engine.name} ready")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "STT load failed", e)
                speechError = e.message ?: e.javaClass.simpleName
            }
        } else {
            Log.w(TAG, "STT ${engine.name} files missing")
        }
        if (_uiState.value.prefs.speakReplies && paths.kokoroReady()) {
            _uiState.value = _uiState.value.copy(status = "Loading Kokoro…")
            try {
                tts.ensureLoaded(paths)
                Log.i(TAG, "Kokoro ready")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Kokoro load failed", e)
                speechError = e.message ?: e.javaClass.simpleName
            }
        }
        return sttReady
    }

    private fun runAutoAsk() {
        val app = getApplication<GemmaVoiceApp>()
        app.dogAsk?.let { request ->
            app.dogAsk = null
            askDog(request)
            return
        }
        val ask = app.autoAsk ?: return
        app.autoAsk = null
        sendText(ask)
    }

    private fun runPointPhoto() {
        val path = getApplication<GemmaVoiceApp>().pointPhoto ?: return
        getApplication<GemmaVoiceApp>().pointPhoto = null
        val photo = java.io.File(path)
        if (!photo.isFile) {
            _uiState.value = _uiState.value.copy(status = "Point photo missing", sceneCaption = "Point photo missing")
            return
        }
        _uiState.value = _uiState.value.copy(cameraOpen = true)
        locatePhoto(photo)
    }

    private suspend fun runVoiceProbe() {
        val wavPath = getApplication<GemmaVoiceApp>().voiceProbeWav ?: return
        getApplication<GemmaVoiceApp>().voiceProbeWav = null
        val wav = java.io.File(wavPath)
        if (!wav.isFile) {
            Log.e(TAG, "PROBE missing $wavPath")
            return
        }
        if (_uiState.value.prefs.voiceStack == VoiceStack.NATIVE && usingGpu) {
            appendUser("Audio file")
            streamAssistant { gpu.generateAudio(wav.absolutePath, _uiState.value.prefs.maxNewTokens) }
            Log.i(TAG, "PROBE audio ${wav.absolutePath}")
            return
        }
        stt.ensureLoaded(paths, _uiState.value.prefs.sttEngine)
        val wave = com.k2fsa.sherpa.onnx.WaveReader.readWave(wav.absolutePath)
        val started = System.nanoTime()
        val transcript = stt.transcribe(wave.samples, wave.sampleRate)
        val sttMs = (System.nanoTime() - started) / 1_000_000
        val heard = transcript.ifBlank { "(empty transcript)" }
        Log.i(TAG, "stt_ms=$sttMs transcript=$heard")
        java.io.File(getApplication<Application>().filesDir, "hear-log.txt")
            .appendText("stt_ms=$sttMs transcript=$heard\n")
        appendUser(heard)
        if (transcript.isBlank()) {
            _uiState.value = _uiState.value.copy(isBusy = false, status = "No speech detected")
            return
        }
        runAssistantGeneration(transcript)
    }

    private fun engineLabel(engine: SttEngine): String = when (engine) {
        SttEngine.SENSE_VOICE -> "SenseVoice"
        SttEngine.WHISPER_TINY_EN -> "Whisper"
    }

    /**
     * Times YOLO and EmbeddingGemma on one photo. Each model is loaded once.
     * The timed loops call the already-open engine and do not construct another.
     */
    private fun runBleProbe() {
        viewModelScope.launch(Dispatchers.IO) {
            val log = java.io.File(getApplication<Application>().filesDir, "play-log.txt")
            _uiState.value = _uiState.value.copy(status = "Looking for a Petoi robot…")
            dog.keepConnected()
            val connected = withTimeoutOrNull(BLE_PROBE_MS) { dog.state.first { it.connected } }
            if (connected != null) {
                log.writeText("bluetooth ${connected.robotName}\nDONE\n")
                _uiState.value = _uiState.value.copy(status = "Bluetooth ${connected.robotName}")
            } else {
                val note = dog.state.value.note
                log.writeText("bluetooth off: $note\nDONE\n")
                _uiState.value = _uiState.value.copy(status = note.ifBlank { "No Bluetooth robot" })
            }
        }
    }

    private fun runTrackProfile(gpu: Boolean) {
        viewModelScope.launch {
            val log = java.io.File(getApplication<Application>().filesDir, "profile-log.txt")
            fun write(line: String) {
                Log.i(TAG, line)
                log.appendText(line + "\n")
                _uiState.value = _uiState.value.copy(isBusy = true, status = line.take(220))
            }
            log.writeText("")
            val photoPath = getApplication<GemmaVoiceApp>().profilePhoto
            getApplication<GemmaVoiceApp>().profilePhoto = null
            val photo = java.io.File(
                photoPath ?: java.io.File(getApplication<Application>().filesDir, "probe/point-scene.jpg").path,
            )
            try {
                if (!photo.isFile) error("Profile photo missing: ${photo.absolutePath}")
                val bitmap = withContext(Dispatchers.Default) {
                    android.graphics.BitmapFactory.decodeFile(photo.absolutePath)
                        ?: error("Could not read ${photo.name}")
                }
                write("photo ${bitmap.width}x${bitmap.height} ${photo.name} gpu=$gpu")
                val yoloLoadStarted = System.nanoTime()
                withContext(Dispatchers.Default) { yolo.ensure(paths.yoloFile()) }
                write("yolo load ${(System.nanoTime() - yoloLoadStarted) / 1_000_000}ms")
                if (!gpu) {
                    val yoloLines = withContext(Dispatchers.Default) {
                        val yoloWarm = timedMs { yolo.detect(bitmap, encodeCrops = false) }
                        val yoloSamples = LongArray(15)
                        var boxes = 0
                        for (index in yoloSamples.indices) {
                            val started = System.nanoTime()
                            boxes = yolo.detect(bitmap, encodeCrops = false).size
                            yoloSamples[index] = (System.nanoTime() - started) / 1_000_000
                        }
                        listOf(
                            "yolo warmup ${yoloWarm}ms",
                            "${summarize("yolo detect", yoloSamples)} boxes=$boxes",
                        )
                    }
                    yoloLines.forEach { write(it) }
                }
                val crops = withContext(Dispatchers.Default) {
                    yolo.detect(bitmap, encodeCrops = true).map { it.jpeg to it.label }
                }
                bitmap.recycle()
                if (crops.isEmpty()) error("YOLO found no objects to embed")
                write(
                    "crops ${crops.size} " + crops.joinToString(",") { (bytes, label) ->
                        "$label:${bytes.size}"
                    },
                )
                val one = listOf(crops.first().first)
                val batch = crops.map { it.first }
                if (gpu) {
                    val float32 = com.google.ai.edge.litertlm.ActivationDataType.FLOAT32
                    val visionOnGpu = timeEmbedBackend(::write, "gpu-f32 vision-gpu", one, batch, useGpu = true, visionGpu = true, activation = float32)
                    if (!visionOnGpu) {
                        timeEmbedBackend(::write, "gpu-f32 vision-cpu", one, batch, useGpu = true, visionGpu = false, activation = float32)
                    }
                } else {
                    timeEmbedBackend(::write, "cpu", one, batch, useGpu = false, visionGpu = false, activation = null)
                }
                write("DONE")
                _uiState.value = _uiState.value.copy(isBusy = false, status = "Profile done")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "profile failed", error)
                write("FAIL ${error.message ?: error.javaClass.simpleName}")
                _uiState.value = _uiState.value.copy(isBusy = false, status = error.message ?: "Profile failed")
            }
        }
    }

    private suspend fun timeEmbedBackend(
        write: (String) -> Unit,
        label: String,
        one: List<ByteArray>,
        batch: List<ByteArray>,
        useGpu: Boolean,
        visionGpu: Boolean,
        activation: com.google.ai.edge.litertlm.ActivationDataType?,
    ): Boolean {
        return try {
            val loadStarted = System.nanoTime()
            embed.open(paths.embeddingVision(), useGpu, visionGpu, activation)
            write("$label load ${(System.nanoTime() - loadStarted) / 1_000_000}ms")
            val oneWarm = timedMs { embed.embedImages(one) }
            write("$label 1 warmup ${oneWarm}ms")
            val oneSamples = LongArray(6)
            for (index in oneSamples.indices) {
                oneSamples[index] = timedMs { embed.embedImages(one) }
            }
            write("$label ${summarize("1 crop", oneSamples)}")
            val batchWarm = timedMs { embed.embedImages(batch) }
            write("$label ${batch.size} warmup ${batchWarm}ms")
            val batchSamples = LongArray(4)
            for (index in batchSamples.indices) {
                batchSamples[index] = timedMs { embed.embedImages(batch) }
            }
            write("$label ${summarize("${batch.size} crops", batchSamples)}")
            val labels = listOf(DOG_QUERY) + POINT_OBJECTS
            val vectors = embed.embedImages(batch)
            val textCold = timedMs { embed.scoreVectors(vectors, labels) }
            val textCached = timedMs { embed.scoreVectors(vectors, labels) }
            write("$label score text cold ${textCold}ms cached ${textCached}ms")
            true
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.e(TAG, "$label failed", error)
            write("$label FAIL ${error.javaClass.simpleName}: ${error.message}")
            false
        }
    }

    private suspend fun timedMs(block: suspend () -> Unit): Long {
        val started = System.nanoTime()
        block()
        return (System.nanoTime() - started) / 1_000_000
    }

    private fun summarize(name: String, samples: LongArray): String {
        val sorted = samples.sorted()
        val mean = samples.sum() / samples.size
        return "$name n=${samples.size} min=${sorted.first()} med=${sorted[sorted.size / 2]} mean=${mean} max=${sorted.last()} ms"
    }

    private suspend fun runBenchIfRequested(prefs: UserPreferences, loadMs: Long) {
        val app = getApplication<GemmaVoiceApp>()
        if (app.benchVariant == null) return
        app.benchVariant = null
        if (usingGpu) {
            Log.i(TAG, "llama bench skipped; chat is on the GPU")
            return
        }
        val report = llm.bench(pp = 512, tg = 128, nr = 3)
        val body = "load_ms=$loadMs\n$report"
        Log.i(TAG, "BENCH ${prefs.gemma.name}\n$body")
        java.io.File(app.filesDir, "bench-${prefs.gemma.name}.txt").writeText(body)
    }

    fun openCamera() {
        _uiState.value = _uiState.value.copy(cameraOpen = true)
        if (_uiState.value.visionReady) return
        loadVision()
    }

    fun closeCamera() {
        stopPlayWatch()
        stopApproach(sendStop = true)
        val running = locateJob
        locateJob = null
        running?.cancel()
        val resumeChat = chatPausedForTrack
        _uiState.value = _uiState.value.copy(
            cameraOpen = false,
            autoScene = false,
            trackLive = false,
            approachLive = false,
            trackBoxes = emptyList(),
            headingLine = emptyList(),
        )
        if (!resumeChat) return
        viewModelScope.launch {
            running?.join()
            withContext(Dispatchers.IO) { embed.close() }
            chatPausedForTrack = false
            loadModels()
        }
    }

    fun setTrackTarget(query: String) {
        _uiState.value = _uiState.value.copy(trackTarget = query)
    }

    fun setTrackLive(enabled: Boolean) {
        if (!enabled) {
            stopPlayWatch()
            _uiState.value = _uiState.value.copy(trackLive = false, sceneCaption = "Live watch stopped.")
            return
        }
        if (_uiState.value.approachLive) {
            stopApproach(sendStop = true)
            _uiState.value = _uiState.value.copy(approachLive = false, headingLine = emptyList())
        }
        _uiState.value = _uiState.value.copy(trackLive = true, cameraOpen = true)
        startPlayWatch()
    }

    fun offerLiveFrame(bitmap: android.graphics.Bitmap) {
        if (!_uiState.value.trackLive && !_uiState.value.approachLive) {
            bitmap.recycle()
            return
        }
        pendingFrame.getAndSet(bitmap)?.recycle()
    }

    /**
     * One camera frame per second. A held object is registered with EmbeddingGemma,
     * and the dog walks only after that same object is still in the person's hands.
     */
    private fun startPlayWatch() {
        if (playJob?.isActive == true) return
        play.reset()
        val log = java.io.File(getApplication<Application>().filesDir, "play-log.txt")
        log.writeText("")
        playJob = viewModelScope.launch {
            try {
                ensurePlayEmbedding()
                withContext(Dispatchers.Default) { yolo.ensure(paths.yoloFile()) }
                dog.keepConnected()
                val link = dog.state.value
                log.appendText(if (link.connected) "bluetooth ${link.robotName}\n" else "bluetooth off: ${link.note}\n")
                while (isActive && _uiState.value.trackLive) {
                    val frame = pendingFrame.getAndSet(null)
                    if (frame == null) {
                        delay(40)
                        continue
                    }
                    val started = System.nanoTime()
                    try {
                        val result = play.step(frame, yolo, embed)
                        val elapsed = (System.nanoTime() - started) / 1_000_000
                        log.appendText("${elapsed}ms ${result.log}\n")
                        _uiState.value = _uiState.value.copy(
                            trackReady = true,
                            sceneCaption = result.caption,
                            status = "${elapsed}ms · ${result.boxes.size} boxes",
                            trackBoxes = result.boxes.map {
                                TrackBox(it.label, it.left, it.top, it.right, it.bottom, it.score, it.role == "dog", it.role)
                            },
                            trackImageWidth = result.width,
                            trackImageHeight = result.height,
                        )
                        val command = result.command
                        if (command != null && dog.isConnected && moveJob?.isActive != true) {
                            moveJob = launch(Dispatchers.IO) {
                                try {
                                    val reply = dog.move(command)
                                    log.appendText("sent $command $reply\n")
                                } catch (cancelled: CancellationException) {
                                    throw cancelled
                                } catch (error: Exception) {
                                    log.appendText("move failed ${error.message}\n")
                                }
                            }
                        }
                        if (result.arrived && !play.announced) {
                            play.announced = true
                            if (dog.isConnected && moveJob?.isActive != true) {
                                moveJob = launch(Dispatchers.IO) {
                                    try {
                                        dog.move("kbalance")
                                    } catch (error: Exception) {
                                        Log.w(TAG, "balance: ${error.message}")
                                    }
                                }
                            }
                            attractAdult()
                        }
                        if (!result.playing) play.announced = false
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (error: Exception) {
                        Log.e(TAG, "play step failed", error)
                        val message = error.message ?: error.javaClass.simpleName
                        log.appendText("step failed $message\n")
                        _uiState.value = _uiState.value.copy(status = message, sceneCaption = message)
                    } finally {
                        frame.recycle()
                    }
                    val spent = (System.nanoTime() - started) / 1_000_000
                    val pause = PLAY_PERIOD_MS - spent
                    if (pause > 0) delay(pause)
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "play watch failed", error)
                _uiState.value = _uiState.value.copy(
                    trackLive = false,
                    status = error.message ?: "Live watch failed",
                    sceneCaption = error.message ?: "Live watch failed",
                )
            }
        }
    }

    private fun stopPlayWatch() {
        playJob?.cancel()
        playJob = null
        moveJob?.cancel()
        moveJob = null
        pendingFrame.getAndSet(null)?.recycle()
        play.reset()
    }

    fun setApproach(enabled: Boolean) {
        if (!enabled) {
            stopApproach(sendStop = true)
            _uiState.value = _uiState.value.copy(approachLive = false, sceneCaption = "Stopped.", headingLine = emptyList())
            return
        }
        if (_uiState.value.trackLive) {
            stopPlayWatch()
            _uiState.value = _uiState.value.copy(trackLive = false)
        }
        _uiState.value = _uiState.value.copy(approachLive = true, cameraOpen = true)
        startApproach()
    }

    /**
     * Look, decide, move, wait for the robot to finish, look again on a frame taken after the
     * move. Repeats until the robot touches the target, the user stops it, or it gives up.
     */
    private fun startApproach() {
        if (approachJob?.isActive == true) return
        pilot.reset()
        val log = java.io.File(getApplication<Application>().filesDir, "approach-log.txt")
        log.writeText("")
        approachJob = viewModelScope.launch {
            try {
                ensurePlayEmbedding()
                withContext(Dispatchers.Default) { yolo.ensure(paths.yoloFile()) }
                dog.keepConnected()
                val query = _uiState.value.trackTarget
                log.appendText("target $query robot ${dog.state.value.robotName.ifBlank { "not connected" }}\n")
                pendingFrame.getAndSet(null)?.recycle()
                while (isActive && _uiState.value.approachLive) {
                    val frame = pendingFrame.getAndSet(null)
                    if (frame == null) {
                        delay(FRAME_POLL_MS)
                        continue
                    }
                    val started = System.nanoTime()
                    val result = try {
                        pilot.step(frame, yolo, embed, query)
                    } finally {
                        frame.recycle()
                    }
                    val elapsed = (System.nanoTime() - started) / 1_000_000
                    log.appendText("${elapsed}ms ${result.log}\n")
                    _uiState.value = _uiState.value.copy(
                        trackReady = true,
                        sceneCaption = result.caption,
                        status = "${elapsed}ms · ${result.boxes.size} boxes",
                        trackBoxes = result.boxes.map {
                            TrackBox(it.label, it.left, it.top, it.right, it.bottom, it.score, it.role == "dog", it.role)
                        },
                        trackImageWidth = result.width,
                        trackImageHeight = result.height,
                        headingLine = result.headingLine,
                    )
                    val command = result.command
                    if (command != null) {
                        if (!dog.isConnected) {
                            _uiState.value = _uiState.value.copy(sceneCaption = "${result.caption} The robot is not connected.")
                            delay(NO_LINK_WAIT_MS)
                            continue
                        }
                        val reply = withContext(Dispatchers.IO) {
                            try {
                                dog.move(command, result.turnHoldMs)
                            } catch (error: IllegalStateException) {
                                "move failed ${error.message}"
                            }
                        }
                        log.appendText("sent $command hold=${result.turnHoldMs} -> $reply\n")
                        // The next decision must see where the robot ended up, not a frame from mid-step.
                        delay(SETTLE_MS)
                        pendingFrame.getAndSet(null)?.recycle()
                    }
                    if (result.arrived || result.gaveUp) {
                        if (result.arrived) attractAdult()
                        _uiState.value = _uiState.value.copy(approachLive = false)
                        break
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "approach failed", error)
                log.appendText("failed ${error.message}\n")
                _uiState.value = _uiState.value.copy(
                    approachLive = false,
                    status = error.message ?: "Walk to target failed",
                    sceneCaption = error.message ?: "Walk to target failed",
                )
            }
        }
    }

    private fun stopApproach(sendStop: Boolean) {
        val running = approachJob?.isActive == true
        approachJob?.cancel()
        approachJob = null
        pendingFrame.getAndSet(null)?.recycle()
        pilot.reset()
        if (sendStop && running && dog.isConnected) {
            // A cancelled forward stride still finishes on the robot; balance makes it stand now.
            viewModelScope.launch(Dispatchers.IO) {
                repeat(STOP_ATTEMPTS) {
                    try {
                        dog.move(ApproachPilot.STOP)
                        return@launch
                    } catch (error: IllegalStateException) {
                        delay(STOP_RETRY_MS)
                    }
                }
            }
        }
    }

    private suspend fun ensurePlayEmbedding() {
        if (chatPausedForTrack && embed.isReady) {
            _uiState.value = _uiState.value.copy(trackReady = true)
            return
        }
        val file = paths.embeddingVision()
        if (!file.isFile) error("Missing ${file.name}. EmbeddingGemma is not on this phone.")
        _uiState.value = _uiState.value.copy(
            status = "Loading EmbeddingGemma on GPU…",
            sceneCaption = "Loading EmbeddingGemma…",
        )
        gpu.close()
        usingGpu = false
        chatPausedForTrack = true
        _uiState.value = _uiState.value.copy(visionReady = false, modelLabel = "Embed GPU")
        try {
            embed.open(file, useGpu = true, visionGpu = true, activation = ActivationDataType.FLOAT32)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (gpuError: Exception) {
            Log.w(TAG, "embedding GPU failed, using CPU", gpuError)
            embed.open(file, useGpu = false)
        }
        _uiState.value = _uiState.value.copy(trackReady = true)
    }

    private fun attractAdult() {
        val app = getApplication<Application>()
        if (AttractSound.play(app)) return
        speakWalk("Come look. The dog found the toy.")
    }

    /**
     * YOLO lists every box. EmbeddingGemma scores those crops against the robot
     * and the object phrases. The dog is asked to step toward that object only
     * when both matches are clearly the best crop.
     */
    fun locatePhoto(jpeg: java.io.File) {
        if (_uiState.value.isBusy || _uiState.value.isRecording || !jpeg.isFile) return
        locateJob?.cancel()
        locateJob = viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isBusy = true,
                status = "Finding objects…",
                sceneCaption = "Finding…",
            )
            try {
                ensureEmbedding()
                val ready = withContext(Dispatchers.IO) { SceneImages.normalize(jpeg) }
                val started = System.nanoTime()
                val bitmap = withContext(Dispatchers.Default) {
                    android.graphics.BitmapFactory.decodeFile(ready.absolutePath)
                        ?: error("Could not read the photo")
                }
                val imageWidth = bitmap.width
                val imageHeight = bitmap.height
                val detections = try {
                    withContext(Dispatchers.Default) {
                        yolo.ensure(paths.yoloFile())
                        yolo.detect(bitmap)
                    }
                } finally {
                    bitmap.recycle()
                }
                if (detections.isEmpty()) error("YOLO found no objects")
                val scores = embed.scoreCrops(
                    detections.map { it.jpeg },
                    listOf(DOG_QUERY) + POINT_OBJECTS,
                )
                val robotScores = scores[DOG_QUERY] ?: error("Robot search returned nothing")
                val robotIndex = bestIndex(robotScores)
                var targetLabel = ""
                var targetIndex = -1
                var targetScore = Float.NEGATIVE_INFINITY
                for (label in POINT_OBJECTS) {
                    val row = scores[label] ?: continue
                    for (index in row.indices) {
                        if (index == robotIndex) continue
                        if (row[index] > targetScore) {
                            targetScore = row[index]
                            targetIndex = index
                            targetLabel = label
                        }
                    }
                }
                if (targetIndex < 0) error("No target box besides the robot")
                val robot = detections[robotIndex]
                val target = detections[targetIndex]
                val robotScore = robotScores[robotIndex]
                val targetRow = scores.getValue(targetLabel)
                val confident = robotScore >= MATCH_FLOOR &&
                    targetScore >= MATCH_FLOOR &&
                    scoreMargin(robotScores, robotIndex) >= MATCH_MARGIN &&
                    scoreMargin(targetRow, targetIndex) >= MATCH_MARGIN
                val dx = target.centerX - robot.centerX
                val dy = target.centerY - robot.centerY
                val skill = if (confident) petoiSkill(dx, dy) else null
                val spokenTarget = shortTarget(targetLabel)
                val moveWords = when (skill) {
                    "kwkR" -> "Go right."
                    "kwkL" -> "Go left."
                    "kvtF" -> "Go forward."
                    "kbkF" -> "Go back."
                    null -> if (confident) "Stop. The robot is at the $spokenTarget." else "Not sure. Not moving."
                    else -> "Hold."
                }
                val walk = if (skill == null && confident) {
                    moveWords
                } else if (!confident) {
                    moveWords
                } else {
                    "Walk to the $spokenTarget. $moveWords"
                }
                val robotNote = sendRobot(skill)
                val elapsed = (System.nanoTime() - started) / 1_000_000
                val boxes = detectionBoxes(detections, robotIndex, targetIndex, spokenTarget, robotScore, targetScore)
                val detail = "yolo ${detections.size} · robot ${scoreText(robotScore)} ${robot.label} · " +
                    "$spokenTarget ${scoreText(targetScore)} · ${elapsed}ms"
                val line = "robot=$robotIndex ${robot.label} ${scoreText(robotScore)} " +
                    "target=$targetIndex $targetLabel ${scoreText(targetScore)} " +
                    "$walk $robotNote ${elapsed}ms boxes=" +
                    detections.joinToString(",") { "${it.label}:${scoreText(it.score)}" }
                Log.i(TAG, line)
                java.io.File(getApplication<Application>().filesDir, "track-log.txt").appendText(line + "\n")
                val previousCommand = _uiState.value.sceneCaption
                val caption = if (robotNote.isBlank()) walk else "$walk $robotNote"
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    trackReady = true,
                    status = detail,
                    sceneCaption = caption,
                    sceneQuestion = "YOLO ${detections.size} boxes. Target $spokenTarget.",
                    scenePhotoPath = ready.absolutePath,
                    trackBoxes = boxes,
                    trackImageWidth = imageWidth,
                    trackImageHeight = imageHeight,
                )
                if (confident && walk != previousCommand) speakWalk(walk)
            } catch (e: CancellationException) {
                _uiState.value = _uiState.value.copy(isBusy = false, status = "Ready")
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "locate failed", e)
                val message = e.message ?: e.javaClass.simpleName
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    trackLive = false,
                    status = message,
                    sceneCaption = message,
                )
            }
        }
    }

    private suspend fun ensureEmbedding() {
        if (embed.isReady) return
        val file = paths.embeddingVision()
        if (!file.isFile) error("Missing ${file.name}. EmbeddingGemma is not on this phone.")
        _uiState.value = _uiState.value.copy(
            status = "Loading EmbeddingGemma…",
            sceneCaption = "Loading EmbeddingGemma…",
        )
        try {
            embed.open(file, useGpu = false)
            Log.i(TAG, "embedding on CPU, Gemma stays loaded")
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (cpuError: Exception) {
            Log.w(TAG, "embedding CPU failed, pausing Gemma for the GPU", cpuError)
            gpu.close()
            usingGpu = false
            chatPausedForTrack = true
            _uiState.value = _uiState.value.copy(visionReady = false, modelLabel = "Embed GPU")
            embed.open(file, useGpu = true)
        }
        _uiState.value = _uiState.value.copy(trackReady = true)
    }

    private suspend fun sendRobot(skill: String?): String {
        if (skill == null) return ""
        val hostFile = paths.robotHostFile()
        if (!hostFile.isFile) return "Robot host is not set."
        val host = hostFile.readLines().firstOrNull { it.isNotBlank() && !it.trim().startsWith("#") }?.trim()
        if (host.isNullOrEmpty()) return "Robot host is not set."
        return withContext(Dispatchers.IO) {
            val reply = petoi.move(host, skill)
            "Sent $skill ($reply)."
        }
    }

    private fun speakWalk(command: String) {
        viewModelScope.launch {
            try {
                if (!paths.kokoroReady()) return@launch
                tts.ensureLoaded(paths)
                tts.speak(command, _uiState.value.prefs.kokoroSpeakerId)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.w(TAG, "walk speech failed: ${error.message}")
            }
        }
    }

    fun setAutoScene(enabled: Boolean) {
        if (!_uiState.value.visionReady) return
        _uiState.value = _uiState.value.copy(autoScene = enabled)
    }

    fun noteSceneError(message: String) {
        _uiState.value = _uiState.value.copy(status = message, sceneCaption = message)
    }

    /**
     * One camera frame. SenseVoice is left in place for the mic.
     * Kokoro is skipped: a spoken caption would block the next frame.
     */
    fun describeScene(jpeg: java.io.File) {
        val state = _uiState.value
        if (!state.visionReady || state.isBusy || !jpeg.isFile) return
        _uiState.value = state.copy(
            isBusy = true,
            status = "Looking at the photo…",
            sceneCaption = "Looking…",
        )
        viewModelScope.launch {
            try {
                val ready = withContext(Dispatchers.IO) { SceneImages.normalize(jpeg) }
                Log.i(TAG, "scene_jpeg bytes=${ready.length()}")
                appendUser("Photo")
                streamAssistant(speak = false, status = "Looking at the photo…", logName = "scene-log.txt") {
                    gpu.generateImage(ready.absolutePath, SCENE_TOKENS)
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "scene prepare failed", e)
                val message = "Photo failed: ${e.message ?: e.javaClass.simpleName}"
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    status = message,
                    sceneCaption = message,
                )
            }
        }
    }

    private fun loadVision() {
        loadJob?.cancel()
        loadJob = viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isBusy = true,
                isReady = false,
                visionReady = false,
                status = "Loading Gemma image model…",
            )
            try {
                val file = paths.litertMultimodal()
                if (!file.isFile) {
                    error("Missing ${file.name}. The image model is not on this phone.")
                }
                val previous = _uiState.value.prefs
                val keepSenseVoice = previous.voiceStack == VoiceStack.PIPELINE && _uiState.value.voiceReady
                var prefs = previous
                if (prefs.gemma != GemmaVariant.E2B || prefs.voiceStack != VoiceStack.PIPELINE) {
                    prefs = prefs.copy(gemma = GemmaVariant.E2B, voiceStack = VoiceStack.PIPELINE)
                    _uiState.value = _uiState.value.copy(prefs = prefs)
                }
                llm.releaseModel()
                val started = System.nanoTime()
                gpu.load(file, prefs.systemPrompt, listen = false, vision = true)
                usingGpu = true
                val loadMs = (System.nanoTime() - started) / 1_000_000
                val line = "LOAD E2B vision ${loadMs}ms"
                Log.i(TAG, line)
                java.io.File(getApplication<Application>().filesDir, "scene-log.txt").appendText(line + "\n")
                val voiceReady = if (keepSenseVoice) true else loadSpeechModels()
                _uiState.value = _uiState.value.copy(
                    isReady = true,
                    isBusy = false,
                    visionReady = true,
                    voiceReady = voiceReady,
                    modelLabel = "E2B vision",
                    status = "Ready",
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "vision load failed", e)
                usingGpu = false
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    isReady = false,
                    visionReady = false,
                    status = "Image model failed: ${e.message ?: e.javaClass.simpleName}",
                    sceneCaption = "Image model failed: ${e.message ?: e.javaClass.simpleName}",
                )
            }
        }
    }

    fun sendText(input: String) {
        val text = input.trim()
        if (text.isEmpty() || !_uiState.value.isReady || _uiState.value.isBusy) return
        appendUser(text)
        runAssistantGeneration(text)
    }

    /** Press on the camera screen: the photo is taken by the preview, speech starts here. */
    fun onDirectionPress() {
        onMicPress()
        if (_uiState.value.isRecording) {
            _uiState.value = _uiState.value.copy(sceneQuestion = "", sceneCaption = "Listening…")
        }
    }

    /**
     * Release: SenseVoice transcribes the utterance, then Gemma sees that text
     * plus the JPEG captured at the press.
     */
    fun onDirectionRelease(photo: CompletableDeferred<java.io.File>?) {
        if (!_uiState.value.isRecording) {
            photo?.cancel()
            return
        }
        recordJob?.cancel()
        val samples = recorder.stopCapture()
        _uiState.value = _uiState.value.copy(
            isRecording = false,
            isBusy = true,
            status = "Transcribing…",
            sceneCaption = "Listening…",
        )
        viewModelScope.launch {
            try {
                stt.ensureLoaded(paths, _uiState.value.prefs.sttEngine)
                val transcript = stt.transcribe(samples, recorder.capturedSampleRate)
                if (transcript.isBlank()) {
                    photo?.cancel()
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        status = "No speech detected",
                        sceneCaption = "No speech detected",
                    )
                    return@launch
                }
                val jpeg = withTimeout(4_000) {
                    photo?.await() ?: error("The photo was not captured")
                }
                val ready = withContext(Dispatchers.IO) { SceneImages.normalize(jpeg) }
                val line = "Q ${transcript.replace('\n', ' ').take(300)}"
                Log.i(TAG, "direction $line jpeg=${ready.length()}")
                java.io.File(getApplication<Application>().filesDir, "scene-log.txt").appendText(line + "\n")
                appendUser(transcript)
                _uiState.value = _uiState.value.copy(
                    sceneQuestion = transcript,
                    sceneCaption = "…",
                    scenePhotoPath = ready.absolutePath,
                )
                streamAssistant(speak = true, status = "Answering…", logName = "scene-log.txt") {
                    gpu.answerImage(ready.absolutePath, transcript, DIRECTION_TOKENS)
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "direction failed", e)
                val message = e.message ?: e.javaClass.simpleName
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    status = message,
                    sceneCaption = message,
                )
            }
        }
    }

    /** Hold on the Dog tab's Speak button. Same recorder as the chat mic. */
    fun onDogMicPress() {
        onMicPress()
        if (_uiState.value.isRecording) {
            _uiState.value = _uiState.value.copy(dogHeard = "", dogSaid = "", dogDid = "", dogTalkNote = "Listening…")
        }
    }

    fun onDogMicRelease() {
        if (!_uiState.value.isRecording) return
        recordJob?.cancel()
        val samples = recorder.stopCapture()
        _uiState.value = _uiState.value.copy(isRecording = false, isBusy = true, dogTalkNote = "Transcribing…")
        viewModelScope.launch {
            try {
                stt.ensureLoaded(paths, _uiState.value.prefs.sttEngine)
                val transcript = stt.transcribe(samples, recorder.capturedSampleRate).trim()
                _uiState.value = _uiState.value.copy(isBusy = false)
                if (transcript.isBlank()) {
                    _uiState.value = _uiState.value.copy(dogTalkNote = "No speech heard. Hold Speak and talk.")
                    return@launch
                }
                askDog(transcript)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "dog listen failed", error)
                _uiState.value = _uiState.value.copy(isBusy = false, dogTalkNote = "Voice error: ${error.message}")
            }
        }
    }

    /**
     * Gemma answers as the dog. The phone speaks the answer (the robot has no speaker) while
     * the robot shows talking eyes and runs the allowed actions, then the eyes go back.
     */
    fun askDog(input: String) {
        val request = input.trim()
        if (request.isEmpty()) return
        val state = _uiState.value
        val blocker = when {
            state.isBusy || state.isRecording -> "Wait for the current answer to finish."
            chatPausedForTrack -> "Close the camera first; Gemma is unloaded while it runs."
            !state.isReady -> "Gemma is still loading."
            !usingGpu -> "Dog speech needs Gemma 4 E2B on the GPU (Settings)."
            else -> null
        }
        if (blocker != null) {
            _uiState.value = state.copy(dogTalkNote = blocker)
            return
        }
        _uiState.value = state.copy(
            isBusy = true,
            dogHeard = request,
            dogSaid = "",
            dogDid = "",
            dogTalkNote = "Thinking…",
        )
        generationJob?.cancel()
        generationJob = viewModelScope.launch {
            val log = java.io.File(getApplication<Application>().filesDir, "dog-talk-log.txt")
            val started = System.nanoTime()
            try {
                val raw = StringBuilder()
                gpu.generateAs(DogSpeech.SYSTEM_PROMPT, request, DOG_TOKENS).collect { raw.append(it) }
                val thinkMs = (System.nanoTime() - started) / 1_000_000
                val reply = DogSpeech.parse(visibleAssistantText(raw.toString()), request)
                val planned = reply.actions.joinToString(" ") { it.command }
                val refused = if (reply.rejected.isEmpty()) "" else "  (refused: ${reply.rejected.joinToString(" ")})"
                log.appendText(
                    "Q $request\nraw ${raw.toString().replace('\n', ' ')}\n" +
                        "say ${reply.say}\ndo ${planned.ifBlank { "-" }}$refused\nthink_ms $thinkMs\n",
                )
                val linked = dog.isConnected
                _uiState.value = _uiState.value.copy(
                    dogSaid = reply.say,
                    dogDid = when {
                        planned.isBlank() -> "No action.$refused"
                        linked -> "$planned$refused"
                        else -> "$planned (robot not connected)$refused"
                    },
                    dogTalkNote = "Speaking…",
                )
                coroutineScope {
                    val body = launch(Dispatchers.IO) {
                        if (!linked) return@launch
                        val lines = dog.perform(listOf(DogSpeech.eyes(DogSpeech.TALK_EYES)) + reply.actions)
                        log.appendText(lines.joinToString("\n", postfix = "\n"))
                    }
                    speakDog(reply.say)
                    body.join()
                }
                if (linked) {
                    withContext(Dispatchers.IO) { dog.perform(listOf(DogSpeech.eyes(DogSpeech.END_TALK_EYES))) }
                }
                _uiState.value = _uiState.value.copy(isBusy = false, dogTalkNote = "Hold Speak to talk to the dog.")
            } catch (cancelled: CancellationException) {
                gpu.cancel()
                _uiState.value = _uiState.value.copy(isBusy = false)
                throw cancelled
            } catch (error: Exception) {
                Log.e(TAG, "dog talk failed", error)
                log.appendText("failed ${error.message}\n")
                _uiState.value = _uiState.value.copy(isBusy = false, dogTalkNote = "Gemma error: ${error.message}")
            }
        }
    }

    private suspend fun speakDog(text: String) {
        if (text.isBlank() || !paths.kokoroReady()) return
        try {
            tts.ensureLoaded(paths)
            val session = tts.openSession(_uiState.value.prefs.kokoroSpeakerId, System.nanoTime())
            for (piece in SpokenChunker().takeReady(text, finished = true)) session.speakPiece(piece)
            session.finish()
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Exception) {
            Log.w(TAG, "dog speech failed: ${error.message}")
        }
    }

    fun onMicPress() {
        if (!_uiState.value.isReady || _uiState.value.isBusy) return
        if (!recorder.start()) {
            _uiState.value = _uiState.value.copy(status = "Microphone unavailable")
            return
        }
        _uiState.value = _uiState.value.copy(isRecording = true, status = "Listening…")
        recordJob = viewModelScope.launch(Dispatchers.IO) {
            while (recorder.isRecording) {
                recorder.pollChunk()
                kotlinx.coroutines.delay(10)
            }
        }
    }

    fun onMicRelease() {
        if (!_uiState.value.isRecording) return
        recordJob?.cancel()
        val samples = recorder.stopCapture()
        val stack = _uiState.value.prefs.voiceStack
        _uiState.value = _uiState.value.copy(
            isRecording = false,
            isBusy = true,
            status = if (stack == VoiceStack.PIPELINE) "Transcribing…" else "Gemma is listening…",
        )
        viewModelScope.launch {
            try {
                if (stack == VoiceStack.NATIVE) {
                    handleNativeClip(samples)
                } else {
                    handlePipelineClip(samples)
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "voice pipeline failed", e)
                _uiState.value = _uiState.value.copy(isBusy = false, status = "Voice error: ${e.message}")
            }
        }
    }

    private suspend fun handlePipelineClip(samples: FloatArray) {
        stt.ensureLoaded(paths, _uiState.value.prefs.sttEngine)
        val transcript = stt.transcribe(samples, recorder.capturedSampleRate)
        if (transcript.isBlank()) {
            _uiState.value = _uiState.value.copy(isBusy = false, status = "No speech detected")
            return
        }
        appendUser(transcript)
        runAssistantGeneration(transcript)
    }

    private suspend fun handleNativeClip(samples: FloatArray) {
        if (!usingGpu) {
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                status = "Gemma audio needs the E2B LiteRT model",
            )
            return
        }
        if (samples.isEmpty()) {
            _uiState.value = _uiState.value.copy(isBusy = false, status = "No speech captured")
            return
        }
        val rate = recorder.capturedSampleRate
        val clipped = if (samples.size > rate * MAX_AUDIO_SECONDS) {
            samples.copyOf(rate * MAX_AUDIO_SECONDS)
        } else {
            samples
        }
        val clip = withContext(Dispatchers.IO) { writeWav(clipped, rate) }
        val seconds = clipped.size / rate.toFloat()
        Log.i(TAG, "audio_wav sec=${"%.1f".format(seconds)} bytes=${clip.length()}")
        appendUser("Spoken ${"%.1f".format(seconds)}s")
        streamAssistant { gpu.generateAudio(clip.absolutePath, _uiState.value.prefs.maxNewTokens) }
    }

    private fun writeWav(samples: FloatArray, sampleRate: Int): java.io.File {
        paths.clipsDir.mkdirs()
        val file = paths.clipsDir.resolve("clip-${System.currentTimeMillis()}.wav")
        val pcm = java.nio.ByteBuffer.allocate(samples.size * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) {
            val clamped = sample.coerceIn(-1f, 1f)
            pcm.putShort((clamped * 32767f).toInt().toShort())
        }
        val data = pcm.array()
        java.io.FileOutputStream(file).use { out ->
            out.write(wavHeader(data.size, sampleRate))
            out.write(data)
        }
        return file
    }

    private fun wavHeader(dataBytes: Int, sampleRate: Int): ByteArray {
        val channels = 1
        val bits = 16
        val byteRate = sampleRate * channels * bits / 8
        val blockAlign = channels * bits / 8
        val chunk = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        chunk.put("RIFF".toByteArray())
        chunk.putInt(36 + dataBytes)
        chunk.put("WAVE".toByteArray())
        chunk.put("fmt ".toByteArray())
        chunk.putInt(16)
        chunk.putShort(1)
        chunk.putShort(channels.toShort())
        chunk.putInt(sampleRate)
        chunk.putInt(byteRate)
        chunk.putShort(blockAlign.toShort())
        chunk.putShort(bits.toShort())
        chunk.put("data".toByteArray())
        chunk.putInt(dataBytes)
        return chunk.array()
    }

    private fun appendUser(text: String) {
        val user = ChatMessage(UUID.randomUUID().toString(), ChatMessage.Role.USER, text)
        _uiState.value = _uiState.value.copy(
            messages = _uiState.value.messages + user,
        )
    }

    private fun runAssistantGeneration(userText: String) {
        val tokens = _uiState.value.prefs.maxNewTokens
        streamAssistant {
            if (usingGpu) gpu.generate(userText, tokens) else llm.generate(userText, tokens)
        }
    }

    private fun streamAssistant(
        speak: Boolean = true,
        status: String = "Generating…",
        logName: String? = null,
        tokens: suspend () -> kotlinx.coroutines.flow.Flow<String>,
    ) {
        generationJob?.cancel()
        val assistantId = UUID.randomUUID().toString()
        val seed = ChatMessage(assistantId, ChatMessage.Role.ASSISTANT, "", isStreaming = true)
        _uiState.value = _uiState.value.copy(
            messages = _uiState.value.messages + seed,
            isBusy = true,
            status = status,
        )
        val builder = StringBuilder()
        val chunker = SpokenChunker()
        generationJob = viewModelScope.launch {
            val started = System.nanoTime()
            val session = if (speak) openSpeechSession(started) else null
            var speechStarted = false
            var completed = false
            try {
                tokens().collect { token ->
                    builder.append(token)
                    val visible = visibleAssistantText(builder.toString())
                    updateAssistant(assistantId, visible, streaming = true)
                    speechStarted = enqueueSpeech(session, chunker, visible, finished = false) || speechStarted
                }
                val reply = visibleAssistantText(builder.toString())
                updateAssistant(assistantId, reply, streaming = false)
                speechStarted = enqueueSpeech(session, chunker, reply, finished = true) || speechStarted
                val genMs = (System.nanoTime() - started) / 1_000_000
                val done = "gen_done_ms=$genMs chars=${reply.length} text=${reply.replace('\n', ' ').take(500)}"
                Log.i(TAG, done)
                val logFile = java.io.File(getApplication<Application>().filesDir, logName ?: "hear-log.txt")
                logFile.appendText(done + "\n")
                if (speechStarted) {
                    _uiState.value = _uiState.value.copy(status = "Speaking…")
                }
                session?.finish()
                completed = true
                _uiState.value = _uiState.value.copy(isBusy = false, status = "Ready")
            } catch (e: CancellationException) {
                if (usingGpu) gpu.cancel()
                session?.stop()
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "generation failed", e)
                session?.stop()
                val message = "LLM error: ${e.message}"
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    status = message,
                    sceneCaption = if (_uiState.value.cameraOpen) message else _uiState.value.sceneCaption,
                )
            } finally {
                if (!completed) session?.stop()
            }
        }
    }

    private suspend fun openSpeechSession(startedNanos: Long): TextToSpeechEngine.SpeechSession? {
        val prefs = _uiState.value.prefs
        if (!prefs.speakReplies || !paths.kokoroReady()) {
            return null
        }
        return try {
            tts.ensureLoaded(paths)
            tts.openSession(prefs.kokoroSpeakerId, startedNanos)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "tts unavailable: ${e.message}")
            null
        }
    }

    private suspend fun enqueueSpeech(
        session: TextToSpeechEngine.SpeechSession?,
        chunker: SpokenChunker,
        visible: String,
        finished: Boolean,
    ): Boolean {
        if (session == null || visible.isEmpty()) return false
        val pieces = chunker.takeReady(visible, finished)
        if (pieces.isEmpty()) return false
        _uiState.value = _uiState.value.copy(status = "Speaking…")
        for (piece in pieces) {
            Log.i(TAG, "speak_chunk chars=${piece.length} text=$piece")
            session.speakPiece(piece)
        }
        return true
    }

    /**
     * Gemma 4 streams a `<|channel>thought` block before the answer.
     * The spoken/shown reply is the text after the last channel closer.
     */
    private fun visibleAssistantText(raw: String): String {
        val thoughtMarker = "<|channel>thought"
        val channelEnd = "<channel|>"
        val body = if (raw.contains(thoughtMarker)) {
            val end = raw.lastIndexOf(channelEnd)
            if (end < 0) return ""
            raw.substring(end + channelEnd.length)
        } else {
            raw
        }
        return body
            .replace("<|turn>model", "")
            .replace("<|channel>response", "")
            .replace("<turn|>", "")
            .trim()
    }

    private fun updateAssistant(id: String, text: String, streaming: Boolean) {
        val camera = _uiState.value.cameraOpen
        _uiState.value = _uiState.value.copy(
            messages = _uiState.value.messages.map {
                if (it.id == id) it.copy(text = text, isStreaming = streaming) else it
            },
            sceneCaption = if (camera && text.isNotEmpty()) text else _uiState.value.sceneCaption,
        )
    }

    override fun onCleared() {
        generationJob?.cancel()
        recordJob?.cancel()
        locateJob?.cancel()
        stopPlayWatch()
        stopApproach(sendStop = false)
        stt.release()
        tts.release()
        embed.close()
        yolo.close()
        gpu.close()
        llm.destroy()
        super.onCleared()
    }

    companion object {
        private const val TAG = "ChatViewModel"
        private const val MAX_AUDIO_SECONDS = 30
        private const val SCENE_TOKENS = 128
        private const val DIRECTION_TOKENS = 160
        private const val PLAY_PERIOD_MS = 1_000L
        private const val BLE_PROBE_MS = 30_000L
        private const val FRAME_POLL_MS = 40L
        private const val DOG_TOKENS = 160
        private const val SETTLE_MS = 400L
        private const val NO_LINK_WAIT_MS = 1_000L
        private const val STOP_ATTEMPTS = 20
        private const val STOP_RETRY_MS = 300L
        private const val DOG_QUERY = "a robot dog"
        private const val MATCH_FLOOR = 0.35f
        private const val MATCH_MARGIN = 0.02f
        private val POINT_OBJECTS = listOf(
            "a water bottle",
            "a cup",
            "a ball",
            "a shoe",
            "a backpack",
            "a chair",
            "a phone",
            "a book",
        )

        private fun scoreText(score: Float): String = String.format(java.util.Locale.US, "%.2f", score)

        private fun bestIndex(scores: FloatArray): Int {
            var best = 0
            for (index in 1 until scores.size) {
                if (scores[index] > scores[best]) best = index
            }
            return best
        }

        private fun scoreMargin(scores: FloatArray, index: Int): Float {
            var second = Float.NEGATIVE_INFINITY
            for (other in scores.indices) {
                if (other != index && scores[other] > second) second = scores[other]
            }
            if (second == Float.NEGATIVE_INFINITY) return scores[index]
            return scores[index] - second
        }

        private fun detectionBoxes(
            detections: List<com.gemmaproj.voicechat.engine.YoloBox>,
            robotIndex: Int,
            targetIndex: Int,
            targetName: String,
            robotScore: Float,
            targetScore: Float,
        ): List<TrackBox> {
            return detections.mapIndexed { index, box ->
                val role = when (index) {
                    robotIndex -> "dog"
                    targetIndex -> "target"
                    else -> "other"
                }
                val label = when (role) {
                    "dog" -> "robot ${scoreText(robotScore)}"
                    "target" -> "$targetName ${scoreText(targetScore)}"
                    else -> box.label
                }
                TrackBox(
                    label = label,
                    left = box.left,
                    top = box.top,
                    right = box.right,
                    bottom = box.bottom,
                    score = box.score,
                    isDog = role == "dog",
                    role = role,
                )
            }
        }

        private fun shortTarget(query: String): String = when (query) {
            TRACK_PERSON -> "person"
            TRACK_BOTTLE -> "bottle"
            else -> query.removePrefix("a ").take(24)
        }

        private fun pointBoxes(fix: com.gemmaproj.voicechat.engine.PointResult): List<TrackBox> {
            val sameCell = fix.dog.col == fix.target.col && fix.dog.row == fix.target.row
            val targetHit = if (sameCell) {
                fix.target.copy(
                    left = fix.target.left + 0.03f,
                    top = fix.target.top + 0.03f,
                    right = fix.target.right - 0.03f,
                    bottom = fix.target.bottom - 0.03f,
                )
            } else {
                fix.target
            }
            return listOf(
                TrackBox(
                    label = "person ${scoreText(fix.person.score)}",
                    left = fix.person.left,
                    top = fix.person.top,
                    right = fix.person.right,
                    bottom = fix.person.bottom,
                    score = fix.person.score,
                    isDog = false,
                    role = "person",
                ),
                TrackBox(
                    label = "dog ${scoreText(fix.dog.score)}",
                    left = fix.dog.left,
                    top = fix.dog.top,
                    right = fix.dog.right,
                    bottom = fix.dog.bottom,
                    score = fix.dog.score,
                    isDog = true,
                    role = "dog",
                ),
                TrackBox(
                    label = "${shortTarget(fix.targetName)} ${scoreText(targetHit.score)}",
                    left = targetHit.left,
                    top = targetHit.top,
                    right = targetHit.right,
                    bottom = targetHit.bottom,
                    score = targetHit.score,
                    isDog = false,
                    role = "target",
                ),
            )
        }
    }
}
