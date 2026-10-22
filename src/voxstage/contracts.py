from __future__ import annotations
from enum import Enum
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]
Seconds = Annotated[FiniteFloat, Field(ge=0, le=300)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tone(str, Enum):
    neutral = "neutral"
    urgent = "urgent"
    gentle = "gentle"
    angry = "angry"
    happy = "happy"
    sad = "sad"


class Character(Strict):
    id: Identifier
    name: str = Field(min_length=1, max_length=40)
    speaker: Literal["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]


class Cue(Strict):
    id: Identifier
    character: Identifier
    text: str = Field(min_length=1, max_length=350)
    speech_text: str | None = Field(default=None, min_length=1, max_length=350)
    language: Literal["Chinese", "English"]
    start: Seconds
    end: Seconds
    tone: Tone = Tone.neutral
    protected_terms: list[str] = Field(default_factory=list,max_length=8)

    @model_validator(mode="after")
    def valid_interval(self):
        if not 0.4 <= self.end-self.start <= 30:
            raise ValueError("cue duration must be between 0.4 and 30 seconds")
        if not any(c.isalnum() for c in self.text):
            raise ValueError("cue must contain spoken content")
        from .text import normalize
        if any(not normalize(t) or normalize(t) not in normalize(self.spoken) for t in self.protected_terms):
            raise ValueError('protected terms must occur in the spoken text')
        return self

    @property
    def spoken(self):
        return self.speech_text or self.text

    @property
    def capacity(self):
        return self.end-self.start


class Policy(Strict):
    max_attempts: int = Field(default=2, ge=1, le=3)
    max_cer: FiniteFloat = Field(default=0.15, ge=0, le=1)
    max_speedup: FiniteFloat = Field(default=1.15, ge=1, le=1.3)
    max_clipped_fraction: FiniteFloat = Field(default=0.001, ge=0, le=0.05)
    min_rms_dbfs: FiniteFloat = Field(default=-48, ge=-80, le=-10)


class Scene(Strict):
    version: Literal[1] = 1
    id: Identifier
    title: str = Field(min_length=1, max_length=100)
    cast: list[Character] = Field(min_length=1, max_length=8)
    cues: list[Cue] = Field(min_length=1, max_length=40)
    policy: Policy = Field(default_factory=Policy)
    seed: int = Field(default=2026, ge=0, le=2**31-1)

    @model_validator(mode="after")
    def validate_graph(self):
        cast_ids = {c.id for c in self.cast}
        if len(cast_ids) != len(self.cast) or len({c.id for c in self.cues}) != len(self.cues):
            raise ValueError("cast and cue IDs must be unique")
        if any(c.character not in cast_ids for c in self.cues):
            raise ValueError("unknown character")
        ordered = sorted(self.cues, key=lambda c:c.start)
        if ordered != self.cues or any(a.end > b.start for a,b in zip(ordered,ordered[1:])):
            raise ValueError("cues must be ordered and non-overlapping")
        return self


class Repair(Strict):
    cue_id: Identifier
    action: Literal["accept", "resynthesize", "faster", "reject"]
    reason: str = Field(min_length=1, max_length=400)


INSTRUCTIONS = {
    Tone.neutral: "清楚、自然地说话，不增加任何台词。Speak clearly and naturally.",
    Tone.urgent: "语气急迫、紧张，保持清晰。Speak urgently with clear articulation.",
    Tone.gentle: "温和、安慰的语气，保持清晰。Speak gently and reassuringly.",
    Tone.angry: "生气但克制地说话，不要尖叫。Sound angry but controlled, without shouting.",
    Tone.happy: "开心、明亮的语气。Speak with a cheerful, bright tone.",
    Tone.sad: "低落而克制的语气，保持清晰。Sound sad but composed and intelligible.",
}
