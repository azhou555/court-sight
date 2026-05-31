"""Broadcast video trimmer: strips between-point dead time using ffmpeg.

Takes a list of PointSegment objects and a source video, extracts each segment
with stream copy (no re-encode), and concatenates them into a single output file.

Usage (programmatic):
    from src.editor.video_editor import VideoEditor
    from src.pipeline.events.boundary_detector import BoundaryDetector

    segments = BoundaryDetector.load("match_boundaries.json")[0]
    editor = VideoEditor()
    editor.trim(
        source="match.mp4",
        segments=segments,
        output="trimmed_match.mp4",
    )

Usage (CLI):
    python -m src.editor.video_editor \\
        --source match.mp4 \\
        --boundaries match_boundaries.json \\
        --output trimmed_match.mp4
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import ffmpeg

from ..pipeline.events.boundary_detector import BoundaryDetector, PointSegment


class VideoEditor:
    """Trims and concatenates point segments from a broadcast match video."""

    def trim(
        self,
        source: str | Path,
        segments: list[PointSegment],
        output: str | Path,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        verbose: bool = False,
    ) -> Path:
        """Extract and concatenate rally segments, stripping dead time.

        Uses ffmpeg stream copy by default (fast, lossless). Pass
        video_codec='libx264' for exact frame-boundary cuts at the cost of
        re-encoding time.

        Args:
            source:      Input video path.
            segments:    Point segments from BoundaryDetector.
            output:      Output video path.
            video_codec: ffmpeg video codec for output ('copy' or encoder name).
            audio_codec: ffmpeg audio codec for output.
            verbose:     Show ffmpeg output.

        Returns:
            Path to the output file.
        """
        source = Path(source)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not segments:
            raise ValueError("No point segments provided.")

        with tempfile.TemporaryDirectory(prefix="court_sight_edit_") as tmp:
            tmp = Path(tmp)
            segment_paths = self._extract_segments(
                source, segments, tmp, video_codec, audio_codec, verbose
            )
            self._concatenate(segment_paths, output, video_codec, audio_codec, verbose)

        print(f"Trimmed video written to {output} ({len(segments)} points)")
        return output

    def _extract_segments(
        self,
        source: Path,
        segments: list[PointSegment],
        tmp_dir: Path,
        video_codec: str,
        audio_codec: str,
        verbose: bool,
    ) -> list[Path]:
        paths = []
        for seg in segments:
            out_path = tmp_dir / f"seg_{seg.point_id:04d}.mp4"
            (
                ffmpeg
                .input(str(source), ss=seg.start_sec, to=seg.end_sec)
                .output(str(out_path), vcodec=video_codec, acodec=audio_codec)
                .overwrite_output()
                .run(quiet=not verbose)
            )
            paths.append(out_path)
            if verbose:
                print(f"  Extracted point {seg.point_id}: {seg.start_sec:.1f}s – {seg.end_sec:.1f}s")
        return paths

    def _concatenate(
        self,
        segment_paths: list[Path],
        output: Path,
        video_codec: str,
        audio_codec: str,
        verbose: bool,
    ) -> None:
        # Write concat manifest
        manifest = output.parent / "_concat_manifest.txt"
        manifest.write_text(
            "\n".join(f"file '{p}'" for p in segment_paths) + "\n"
        )
        try:
            (
                ffmpeg
                .input(str(manifest), format="concat", safe=0)
                .output(str(output), vcodec=video_codec, acodec=audio_codec)
                .overwrite_output()
                .run(quiet=not verbose)
            )
        finally:
            manifest.unlink(missing_ok=True)


def trim_from_boundaries(
    source: str | Path,
    boundaries_json: str | Path,
    output: str | Path,
    video_codec: str = "copy",
    verbose: bool = False,
) -> Path:
    """Convenience wrapper: load boundaries JSON and trim in one call."""
    segments, _ = BoundaryDetector.load(boundaries_json)
    return VideoEditor().trim(
        source=source,
        segments=segments,
        output=output,
        video_codec=video_codec,
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trim broadcast match video to rally segments only")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--boundaries", required=True, help="match_boundaries.json from BoundaryDetector")
    parser.add_argument("--output", required=True, help="Output trimmed video path")
    parser.add_argument("--codec", default="copy", help="Video codec (default: copy)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    trim_from_boundaries(
        source=args.source,
        boundaries_json=args.boundaries,
        output=args.output,
        video_codec=args.codec,
        verbose=args.verbose,
    )
