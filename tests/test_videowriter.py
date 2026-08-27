from types import SimpleNamespace

from MetLib.metstruct import ExportOption, FFMpegConfig
import MetLib.videowriter as videowriter


class FakeVideoLoader:
    fps = 10.0
    video_name = "source.mp4"
    start_frame = 100
    end_frame = 126


def _setup_writer(monkeypatch, audio_codec):
    config = FFMpegConfig(path=None,
                          preset="fast",
                          crf=20,
                          ffmpeg_path="ffmpeg",
                          ffprobe_path="ffprobe")
    option = ExportOption(ffmpeg_config=config)
    commands = []

    monkeypatch.setattr(videowriter, "VanillaVideoLoader", FakeVideoLoader)
    monkeypatch.setattr(
        videowriter.PyAVVideoWriter, "save_tmp_avi",
        classmethod(lambda cls, *args, **kwargs: "temporary.avi"))
    monkeypatch.setattr(
        videowriter.FFMpegVideoWriter, "_chk_ffmpeg_path",
        classmethod(lambda cls, candidate: candidate))
    monkeypatch.setattr(
        videowriter.FFMpegVideoWriter, "_probe_audio_codec",
        classmethod(lambda cls, ffprobe, src: audio_codec))

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(videowriter.subprocess, "run", fake_run)
    return option, commands


def test_save_video_with_audio_resets_pts_and_reencodes_audio(monkeypatch):
    option, commands = _setup_writer(monkeypatch, "aac")

    status = videowriter.FFMpegVideoWriter.save_video_with_audio(
        [], FakeVideoLoader(), option, "clip.mp4")

    assert status == 0
    command = commands[-1]
    filter_value = command[command.index("-filter_complex") + 1]
    assert "setpts=PTS-STARTPTS" in filter_value
    assert "atrim=duration=2.600000" in filter_value
    assert "asetpts=PTS-STARTPTS" in filter_value
    assert command[command.index("-c:a") + 1] == "aac"
    assert "copy" not in command
    assert "-shortest" in command
    assert command[-4:-2] == ["-t", "2.600000"]


def test_save_video_with_audio_tolerates_source_without_audio(monkeypatch):
    option, commands = _setup_writer(monkeypatch, None)

    status = videowriter.FFMpegVideoWriter.save_video_with_audio(
        [], FakeVideoLoader(), option, "clip.mp4")

    assert status == 0
    command = commands[-1]
    assert command.count("-i") == 1
    assert "-an" in command
    assert "setpts=PTS-STARTPTS" in command

