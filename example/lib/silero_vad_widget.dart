import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:fonnx/models/sileroVad/silero_vad.dart';
import 'package:fonnx_example/padding.dart';
import 'dart:async';
import 'package:path_provider/path_provider.dart' as path_provider;
import 'package:path/path.dart' as path;
import 'package:record/record.dart';

class SileroVadWidget extends StatefulWidget {
  const SileroVadWidget({super.key});

  @override
  State<SileroVadWidget> createState() => _SileroVadWidgetState();
}

class _SileroVadWidgetState extends State<SileroVadWidget> {
  bool? _verifyPassed;
  String? _speedTestResult;

  // ===== Live VAD demo state =====
  bool _isLiveDemoRunning = false;
  double? _currentVadP; // Latest VAD probability (0..1)
  bool _speechDetected = false;
  bool _speechDetectedLast5s = false;
  StreamSubscription<Uint8List>? _micSub;
  AudioRecorder? _audioRecorder;
  SileroVad? _liveVad;
  final List<int> _micBuffer = [];
  Map<String, dynamic> _lastVadState = {};
  final List<_VadSample> _recentVad = [];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        heightPadding,
        Text(
          'Silero VAD',
          style: Theme.of(context).textTheme.headlineLarge,
        ),
        const Text(
            '1 MB model detects when speech is present in audio. By Silero.'),
        heightPadding,
        Row(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            ElevatedButton(
              onPressed: _runVerificationTest,
              child: const Text('Test Correctness'),
            ),
            widthPadding,
            if (_verifyPassed == true)
              const Icon(
                Icons.check,
                color: Colors.green,
              ),
            if (_verifyPassed == false)
              const Icon(
                Icons.close,
                color: Colors.red,
              ),
          ],
        ),
        heightPadding,
        Row(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            ElevatedButton(
              onPressed: _runPerformanceTest,
              child: const Text('Test Speed'),
            ),
            widthPadding,
            if (_speedTestResult != null)
              Text(
                '${_speedTestResult}x realtime',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
          ],
        ),
        heightPadding,
        _buildLiveDemoSection(context),
      ],
    );
  }

  Widget _buildLiveDemoSection(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: _toggleLiveDemo,
                  icon:
                      Icon(_isLiveDemoRunning ? Icons.stop : Icons.play_arrow),
                  label: Text(_isLiveDemoRunning
                      ? 'Stop Live Demo'
                      : 'Start Live Demo'),
                ),
                const SizedBox(width: 12),
                if (_isLiveDemoRunning)
                  Row(
                    children: [
                      Icon(
                        _speechDetected ? Icons.mic : Icons.mic_off,
                        color: _speechDetected ? Colors.red : Colors.grey,
                        size: 28,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        _speechDetected ? 'Speech' : 'Silence',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ],
                  ),
              ],
            ),
            if (_isLiveDemoRunning) ...[
              const SizedBox(height: 12),
              LinearProgressIndicator(
                minHeight: 8,
                value: _currentVadP?.clamp(0.0, 1.0) ?? 0.0,
                backgroundColor: Colors.grey.shade300,
                valueColor: AlwaysStoppedAnimation<Color>(
                  _speechDetected ? Colors.red : Colors.grey,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                  'VAD probability: ${_currentVadP != null ? _currentVadP!.toStringAsFixed(2) : '--'}'),
              const SizedBox(height: 4),
              Row(
                children: [
                  Icon(
                      _speechDetectedLast5s
                          ? Icons.graphic_eq
                          : Icons.hearing_disabled,
                      color:
                          _speechDetectedLast5s ? Colors.green : Colors.grey),
                  const SizedBox(width: 8),
                  Text(_speechDetectedLast5s
                      ? 'Speech present in last 5 seconds'
                      : 'No speech in last 5 seconds'),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  void _runVerificationTest() async {
    final modelPath = await getModelPath('silero_vad.onnx');
    final silero = SileroVad.load(modelPath);
    final wavFile = await rootBundle.load('assets/audio_sample_16khz.wav');
    final result = await silero.doInference(wavFile.buffer.asUint8List());
    setState(() {
      // obtained on macOS M2 9 Feb 2024.
      final acceptableAnswers = {
        0.4739372134208679, // macOS MBP M2 10 Feb 2024
        0.4739373028278351, // Android Pixel Fold 10 Feb 2024
        0.4739360809326172, // Web 15 Feb 2024
      };
      _verifyPassed = result.length == 3 &&
          acceptableAnswers.contains(result['output'].first);
      if (_verifyPassed != true) {
        if (kDebugMode) {
          print(
              'verification of Silero output failed, got ${result['output']}');
        }
      }
    });
  }

  void _runPerformanceTest() async {
    final modelPath = await getModelPath('silero_vad.onnx');
    final sileroVad = SileroVad.load(modelPath);
    final result = await testPerformance(sileroVad);
    setState(() {
      _speedTestResult = result;
    });
  }

  static Future<String> testPerformance(SileroVad sileroVad) async {
    final vadPerfWavFile =
        await rootBundle.load('assets/audio_sample_16khz.wav');
    final bytes = vadPerfWavFile.buffer.asUint8List();
    const iterations = 3;
    final Stopwatch sw = Stopwatch();
    for (var i = 0; i < iterations; i++) {
      if (i == 1) {
        sw.start();
      }
      await sileroVad.doInference(bytes);
    }
    sw.stop();
    debugPrint('Silero VAD performance:');
    final average =
        sw.elapsedMilliseconds.toDouble() / (iterations - 1).toDouble();
    debugPrint('  Average: ${average.toStringAsFixed(0)} ms');
    debugPrint('  Total: ${sw.elapsedMilliseconds} ms');
    const fileDurationMs = 5000;
    final speedMultilper = fileDurationMs.toDouble() / average;
    debugPrint('  Speed multiplier: ${speedMultilper.toStringAsFixed(2)}x');
    debugPrint('  Model path: ${sileroVad.modelPath}');
    return speedMultilper.toStringAsFixed(2);
  }

  Future<String> getModelPath(String modelFilenameWithExtension) async {
    if (kIsWeb) {
      return 'assets/models/sileroVad/$modelFilenameWithExtension';
    }
    final assetCacheDirectory =
        await path_provider.getApplicationSupportDirectory();
    final modelPath =
        path.join(assetCacheDirectory.path, modelFilenameWithExtension);

    File file = File(modelPath);
    bool fileExists = await file.exists();
    final fileLength = fileExists ? await file.length() : 0;

    // Do not use path package / path.join for paths.
    // After testing on Windows, it appears that asset paths are _always_ Unix style, i.e.
    // use /, but path.join uses \ on Windows.
    final assetPath =
        'assets/models/sileroVad/${path.basename(modelFilenameWithExtension)}';
    final assetByteData = await rootBundle.load(assetPath);
    final assetLength = assetByteData.lengthInBytes;
    final fileSameSize = fileLength == assetLength;
    if (!fileExists || !fileSameSize) {
      debugPrint(
          'Copying model to $modelPath. Why? Either the file does not exist (${!fileExists}), '
          'or it does exist but is not the same size as the one in the assets '
          'directory. (${!fileSameSize})');
      debugPrint('About to get byte data for $modelPath');

      List<int> bytes = assetByteData.buffer.asUint8List(
        assetByteData.offsetInBytes,
        assetByteData.lengthInBytes,
      );
      debugPrint('About to copy model to $modelPath');
      try {
        if (!fileExists) {
          await file.create(recursive: true);
        }
        await file.writeAsBytes(bytes, flush: true);
      } catch (e) {
        debugPrint('Error writing bytes to $modelPath: $e');
        rethrow;
      }
      debugPrint('Copied model to $modelPath');
    }

    return modelPath;
  }

  // ================= Live demo logic =================
  static const int _sampleRate = 16000;
  static const int _channels = 1;
  static const int _bitsPerSample = 16; // pcm16
  static const int _frameMs = 30;
  static final int _frameSizeBytes =
      _sampleRate * _frameMs * _channels * (_bitsPerSample ~/ 8) ~/ 1000;

  Future<void> _toggleLiveDemo() async {
    if (_isLiveDemoRunning) {
      await _stopLiveDemo();
    } else {
      await _startLiveDemo();
    }
  }

  Future<void> _startLiveDemo() async {
    final hasPermission = await AudioRecorder().hasPermission();
    if (!hasPermission) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Microphone permission denied')),
      );
      return;
    }

    final modelPath = await getModelPath('silero_vad.onnx');
    _liveVad = SileroVad.load(modelPath);

    _audioRecorder = AudioRecorder();
    final stream = await _audioRecorder!.startStream(
      const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        numChannels: _channels,
        sampleRate: _sampleRate,
        echoCancel: false,
        noiseSuppress: false,
      ),
    );

    _isLiveDemoRunning = true;
    setState(() {});

    _micSub = stream.listen((event) async {
      _micBuffer.addAll(event);
      while (_micBuffer.length >= _frameSizeBytes) {
        final frameBytes =
            Uint8List.fromList(_micBuffer.sublist(0, _frameSizeBytes));
        _micBuffer.removeRange(0, _frameSizeBytes);

        if (_liveVad == null) continue;
        final nextState = await _liveVad!
            .doInference(frameBytes, previousState: _lastVadState);
        _lastVadState = nextState;
        final p = (nextState['output'] as Float32List).first;

        // Maintain 5-second rolling history
        _recentVad.add(_VadSample(p));
        final cutoff = DateTime.now().subtract(const Duration(seconds: 5));
        while (_recentVad.isNotEmpty && _recentVad.first.ts.isBefore(cutoff)) {
          _recentVad.removeAt(0);
        }
        _speechDetectedLast5s = _recentVad.any((s) => s.p >= 0.5);

        setState(() {
          _currentVadP = p;
          _speechDetected = p >= 0.5;
        });
      }
    });
  }

  Future<void> _stopLiveDemo() async {
    _isLiveDemoRunning = false;
    _currentVadP = null;
    _speechDetected = false;
    _speechDetectedLast5s = false;
    _lastVadState = {};
    _micBuffer.clear();
    await _micSub?.cancel();
    _micSub = null;
    if (_audioRecorder != null) {
      if (await _audioRecorder!.isRecording()) {
        await _audioRecorder!.stop();
      }
      _audioRecorder = null;
    }
    _liveVad = null;
    _recentVad.clear();
    if (mounted) {
      setState(() {});
    }
  }

  @override
  void dispose() {
    _stopLiveDemo();
    super.dispose();
  }
}

// Holder for VAD probability with timestamp
class _VadSample {
  final double p;
  final DateTime ts;

  _VadSample(this.p) : ts = DateTime.now();
}