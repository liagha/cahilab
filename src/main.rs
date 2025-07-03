use {
    cpal::{
        traits::{DeviceTrait, HostTrait, StreamTrait},
        Stream, StreamConfig,
    },
    std::{
        time::Instant,
        sync::{Arc, Mutex},
        collections::HashMap,
    }
};

pub type Sample = f32;
pub type Rate = u32;
pub type Frame = Vec<Sample>;

#[derive(Clone)]
pub struct Buffer {
    pub data: Vec<Sample>,
    pub rate: Rate,
    pub channels: usize,
}

impl Buffer {
    pub fn new(size: usize, rate: Rate, channels: usize) -> Self {
        Self {
            data: vec![0.0; size * channels],
            rate,
            channels,
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    pub fn mix(&mut self, other: &Buffer) {
        for (i, &sample) in other.data.iter().enumerate() {
            if i < self.data.len() {
                self.data[i] += sample;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Wave {
    Sine,
    Square,
    Sawtooth,
    Triangle,
    Noise,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    LowPass,
    HighPass,
    BitCrush,
}

#[derive(Debug, Clone, Copy)]
pub struct Pitch {
    pub value: f32,
    pub velocity: f32,
    pub length: f32,
    pub start: f32,
}

impl Pitch {
    pub fn from_midi(note: u8, velocity: f32, length: f32, start: f32) -> Self {
        let value = 440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0);
        Self { value, velocity, length, start }
    }
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
}

impl Shape {
    pub fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        Self { attack, decay, sustain, release }
    }

    pub fn apply(&self, time: f32, duration: f32) -> f32 {
        match time {
            t if t < self.attack => t / self.attack,
            t if t < self.attack + self.decay => {
                1.0 - (1.0 - self.sustain) * (t - self.attack) / self.decay
            }
            t if t < duration - self.release => self.sustain,
            t if t < duration => self.sustain * (duration - t) / self.release,
            _ => 0.0,
        }
    }
}

pub trait Generator: Send + Sync {
    fn next(&mut self, frequency: f32, time: f32) -> Sample;
    fn reset(&mut self);
    fn clone_box(&self) -> Box<dyn Generator>;
}

pub trait Processor: Send + Sync {
    fn process(&mut self, input: Sample) -> Sample;
    fn reset(&mut self);
    fn clone_box(&self) -> Box<dyn Processor>;
}

pub trait Device: Send + Sync {
    fn render(&mut self, buffer: &mut Buffer);
    fn trigger(&mut self, pitch: Pitch);
    fn set_volume(&mut self, volume: f32);
}

#[derive(Clone)]
pub struct Sine {
    phase: f32,
    rate: Rate,
}

impl Sine {
    pub fn new(rate: Rate) -> Self {
        Self { phase: 0.0, rate }
    }
}

impl Generator for Sine {
    fn next(&mut self, frequency: f32, _time: f32) -> Sample {
        let sample = (self.phase * 2.0 * std::f32::consts::PI).sin();
        self.phase += frequency / self.rate as f32;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Generator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Square {
    phase: f32,
    rate: Rate,
    width: f32,
}

impl Square {
    pub fn new(rate: Rate, width: f32) -> Self {
        Self { phase: 0.0, rate, width }
    }
}

impl Generator for Square {
    fn next(&mut self, frequency: f32, _time: f32) -> Sample {
        let sample = if self.phase < self.width { 1.0 } else { -1.0 };
        self.phase += frequency / self.rate as f32;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Generator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Sawtooth {
    phase: f32,
    rate: Rate,
}

impl Sawtooth {
    pub fn new(rate: Rate) -> Self {
        Self { phase: 0.0, rate }
    }
}

impl Generator for Sawtooth {
    fn next(&mut self, frequency: f32, _time: f32) -> Sample {
        let sample = 2.0 * self.phase - 1.0;
        self.phase += frequency / self.rate as f32;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Generator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Triangle {
    phase: f32,
    rate: Rate,
}

impl Triangle {
    pub fn new(rate: Rate) -> Self {
        Self { phase: 0.0, rate }
    }
}

impl Generator for Triangle {
    fn next(&mut self, frequency: f32, _time: f32) -> Sample {
        let sample = if self.phase < 0.5 {
            4.0 * self.phase - 1.0
        } else {
            3.0 - 4.0 * self.phase
        };
        self.phase += frequency / self.rate as f32;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        sample
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Generator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Noise {
    state: u32,
}

impl Noise {
    pub fn new() -> Self {
        Self { state: 0xACE1 }
    }
}

impl Generator for Noise {
    fn next(&mut self, _frequency: f32, _time: f32) -> Sample {
        let bit = ((self.state >> 14) ^ (self.state >> 13)) & 1;
        self.state = (self.state << 1) | bit;
        self.state &= 0x7FFF;
        if self.state & 1 == 1 { 1.0 } else { -1.0 }
    }

    fn reset(&mut self) {
        self.state = 0xACE1;
    }

    fn clone_box(&self) -> Box<dyn Generator> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Low {
    cutoff: f32,
    rate: Rate,
    prev: Sample,
}

impl Low {
    pub fn new(cutoff: f32, rate: Rate) -> Self {
        Self { cutoff, rate, prev: 0.0 }
    }
}

impl Processor for Low {
    fn process(&mut self, input: Sample) -> Sample {
        let alpha = 1.0 / (1.0 + self.rate as f32 / (2.0 * std::f32::consts::PI * self.cutoff));
        self.prev = alpha * input + (1.0 - alpha) * self.prev;
        self.prev
    }

    fn reset(&mut self) {
        self.prev = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Processor> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct High {
    cutoff: f32,
    rate: Rate,
    prev_input: Sample,
    prev_output: Sample,
}

impl High {
    pub fn new(cutoff: f32, rate: Rate) -> Self {
        Self {
            cutoff,
            rate,
            prev_input: 0.0,
            prev_output: 0.0,
        }
    }
}

impl Processor for High {
    fn process(&mut self, input: Sample) -> Sample {
        let alpha = 1.0 / (1.0 + self.rate as f32 / (2.0 * std::f32::consts::PI * self.cutoff));
        let output = alpha * (self.prev_output + input - self.prev_input);
        self.prev_input = input;
        self.prev_output = output;
        output
    }

    fn reset(&mut self) {
        self.prev_input = 0.0;
        self.prev_output = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Processor> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
pub struct Crush {
    bits: u32,
    reduction: u32,
    counter: u32,
    last: Sample,
}

impl Crush {
    pub fn new(bits: u32, reduction: u32) -> Self {
        Self { bits, reduction, counter: 0, last: 0.0 }
    }
}

impl Processor for Crush {
    fn process(&mut self, input: Sample) -> Sample {
        self.counter += 1;
        if self.counter >= self.reduction {
            self.counter = 0;
            let max_value = (1 << (self.bits - 1)) as f32;
            let quantized = (input * max_value).round() / max_value;
            self.last = quantized.clamp(-1.0, 1.0);
        }
        self.last
    }

    fn reset(&mut self) {
        self.counter = 0;
        self.last = 0.0;
    }

    fn clone_box(&self) -> Box<dyn Processor> {
        Box::new(self.clone())
    }
}

pub struct Voice {
    generator: Box<dyn Generator>,
    shape: Shape,
    processors: Vec<Box<dyn Processor>>,
    pitch: Option<Pitch>,
    start: Instant,
    active: bool,
}

impl Voice {
    pub fn new(generator: Box<dyn Generator>, shape: Shape) -> Self {
        Self {
            generator,
            shape,
            processors: Vec::new(),
            pitch: None,
            start: Instant::now(),
            active: false,
        }
    }

    pub fn add_processor(&mut self, processor: Box<dyn Processor>) {
        self.processors.push(processor);
    }

    pub fn trigger(&mut self, pitch: Pitch) {
        self.pitch = Some(pitch);
        self.start = Instant::now();
        self.active = true;
        self.generator.reset();
        for processor in &mut self.processors {
            processor.reset();
        }
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    pub fn render(&mut self, _rate: Rate) -> Sample {
        if !self.active || self.pitch.is_none() {
            return 0.0;
        }

        let pitch = self.pitch.unwrap();
        let elapsed = self.start.elapsed().as_secs_f32();

        if elapsed > pitch.length {
            self.active = false;
            return 0.0;
        }

        let mut sample = self.generator.next(pitch.value, elapsed);
        let envelope = self.shape.apply(elapsed, pitch.length);
        sample *= envelope * pitch.velocity;

        for processor in &mut self.processors {
            sample = processor.process(sample);
        }

        sample
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

pub struct Instrument {
    voices: Vec<Voice>,
    rate: Rate,
    volume: f32,
}

impl Instrument {
    pub fn new(rate: Rate) -> Self {
        Self {
            voices: Vec::new(),
            rate,
            volume: 1.0,
        }
    }

    pub fn add_voice(&mut self, voice: Voice) {
        self.voices.push(voice);
    }
}

impl Device for Instrument {
    fn render(&mut self, buffer: &mut Buffer) {
        buffer.clear();
        let samples_per_channel = buffer.data.len() / buffer.channels;

        for i in 0..samples_per_channel {
            let mut mixed = 0.0;
            for voice in &mut self.voices {
                mixed += voice.render(self.rate);
            }
            mixed *= self.volume;
            mixed = mixed.clamp(-1.0, 1.0);

            for ch in 0..buffer.channels {
                buffer.data[i * buffer.channels + ch] = mixed;
            }
        }
    }

    fn trigger(&mut self, pitch: Pitch) {
        for voice in &mut self.voices {
            if !voice.is_active() {
                voice.trigger(pitch);
                break;
            }
        }
    }

    fn set_volume(&mut self, volume: f32) {
        self.volume = volume;
    }
}

#[derive(Clone)]
pub struct Sequence {
    pub notes: Vec<Pitch>,
    pub length: f32,
    pub tempo: f32,
}

impl Sequence {
    pub fn new(length: f32, tempo: f32) -> Self {
        Self { notes: Vec::new(), length, tempo }
    }

    pub fn add(&mut self, pitch: Pitch) {
        self.notes.push(pitch);
    }

    pub fn get_range(&self, start: f32, end: f32) -> Vec<Pitch> {
        let pattern_start = start % self.length;
        let pattern_end = end % self.length;

        self.notes.iter()
            .filter(|pitch| {
                let pitch_start = pitch.start % self.length;
                if pattern_start <= pattern_end {
                    pitch_start >= pattern_start && pitch_start < pattern_end
                } else {
                    pitch_start >= pattern_start || pitch_start < pattern_end
                }
            })
            .cloned()
            .collect()
    }
}

pub struct Timeline {
    sequences: HashMap<String, Sequence>,
    current: Option<String>,
    playing: bool,
    start: Instant,
    tempo: f32,
    last_time: f32,
}

impl Timeline {
    pub fn new(tempo: f32) -> Self {
        Self {
            sequences: HashMap::new(),
            current: None,
            playing: false,
            start: Instant::now(),
            tempo,
            last_time: 0.0,
        }
    }

    pub fn add_sequence(&mut self, name: String, sequence: Sequence) {
        self.sequences.insert(name, sequence);
    }

    pub fn play(&mut self, name: &str) {
        if self.sequences.contains_key(name) {
            self.current = Some(name.to_string());
            self.playing = true;
            self.start = Instant::now();
            self.last_time = 0.0;
        }
    }

    pub fn tick(&mut self, delta: f32) -> Vec<Pitch> {
        if !self.playing || self.current.is_none() {
            return Vec::new();
        }

        let name = self.current.as_ref().unwrap();
        if let Some(sequence) = self.sequences.get(name) {
            let current_time = self.start.elapsed().as_secs_f32();
            let notes = sequence.get_range(self.last_time, current_time + delta);
            self.last_time = current_time + delta;
            notes
        } else {
            Vec::new()
        }
    }

    pub fn start(&mut self) {
        self.playing = true;
    }

    pub fn stop(&mut self) {
        self.playing = false;
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }
}

pub struct Studio {
    devices: Vec<Arc<Mutex<dyn Device>>>,
    timeline: Arc<Mutex<Timeline>>,
    rate: Rate,
    buffer_size: usize,
    stream: Option<Stream>,
}

impl Studio {
    pub fn new(rate: Rate, buffer_size: usize) -> Self {
        Self {
            devices: Vec::new(),
            timeline: Arc::new(Mutex::new(Timeline::new(120.0))),
            rate,
            buffer_size,
            stream: None,
        }
    }

    pub fn add_device(&mut self, device: impl Device + 'static) {
        self.devices.push(Arc::new(Mutex::new(device)));
    }

    pub fn get_timeline(&self) -> Arc<Mutex<Timeline>> {
        self.timeline.clone()
    }

    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or("No output device available")?;

        let config = StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(self.rate),
            buffer_size: cpal::BufferSize::Fixed(self.buffer_size as u32),
        };

        let devices = self.devices.clone();
        let timeline = self.timeline.clone();
        let rate = self.rate;
        let buffer_size = self.buffer_size;

        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let delta = buffer_size as f32 / rate as f32;
                let mut buffer = Buffer::new(buffer_size, rate, 2);

                if let Ok(mut timeline) = timeline.lock() {
                    let pitches = timeline.tick(delta);
                    drop(timeline);

                    for pitch in pitches {
                        for device_mutex in &devices {
                            if let Ok(mut device) = device_mutex.lock() {
                                device.trigger(pitch);
                                break;
                            }
                        }
                    }
                }

                for device_mutex in &devices {
                    if let Ok(mut device) = device_mutex.lock() {
                        let mut device_buffer = Buffer::new(buffer_size, rate, 2);
                        device.render(&mut device_buffer);
                        buffer.mix(&device_buffer);
                    }
                }

                for (i, sample) in data.iter_mut().enumerate() {
                    *sample = if i < buffer.data.len() {
                        buffer.data[i]
                    } else {
                        0.0
                    };
                }
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )?;

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }
    }
}

pub struct Builder;

impl Builder {
    pub fn generator(wave: Wave, rate: Rate) -> Box<dyn Generator> {
        match wave {
            Wave::Sine => Box::new(Sine::new(rate)),
            Wave::Square => Box::new(Square::new(rate, 0.5)),
            Wave::Sawtooth => Box::new(Sawtooth::new(rate)),
            Wave::Triangle => Box::new(Triangle::new(rate)),
            Wave::Noise => Box::new(Noise::new()),
        }
    }

    pub fn processor(filter: FilterType, rate: Rate) -> Box<dyn Processor> {
        match filter {
            FilterType::LowPass => Box::new(Low::new(1000.0, rate)),
            FilterType::HighPass => Box::new(High::new(200.0, rate)),
            FilterType::BitCrush => Box::new(Crush::new(6, 2)),
        }
    }

    pub fn voice(wave: Wave, rate: Rate) -> Voice {
        Voice::new(
            Self::generator(wave, rate),
            Shape::new(0.01, 0.1, 0.8, 0.1)
        )
    }

    pub fn melody_synth(rate: Rate) -> Instrument {
        let mut synth = Instrument::new(rate);
        synth.set_volume(0.3);

        for _ in 0..4 {
            let mut voice = Self::voice(Wave::Square, rate);
            voice.add_processor(Self::processor(FilterType::BitCrush, rate));
            synth.add_voice(voice);
        }
        synth
    }

    pub fn bass_synth(rate: Rate) -> Instrument {
        let mut synth = Instrument::new(rate);
        synth.set_volume(0.4);

        for _ in 0..2 {
            let mut voice = Voice::new(
                Box::new(Square::new(rate, 0.25)),
                Shape::new(0.01, 0.05, 0.9, 0.05)
            );
            voice.add_processor(Box::new(Crush::new(4, 3)));
            synth.add_voice(voice);
        }
        synth
    }

    pub fn arp_synth(rate: Rate) -> Instrument {
        let mut synth = Instrument::new(rate);
        synth.set_volume(0.2);

        for _ in 0..3 {
            let mut voice = Voice::new(
                Self::generator(Wave::Triangle, rate),
                Shape::new(0.01, 0.2, 0.3, 0.2)
            );
            voice.add_processor(Box::new(Crush::new(5, 2)));
            synth.add_voice(voice);
        }
        synth
    }

    pub fn drum_synth(rate: Rate) -> Instrument {
        let mut synth = Instrument::new(rate);
        synth.set_volume(0.5);

        for _ in 0..2 {
            let mut voice = Voice::new(
                Self::generator(Wave::Noise, rate),
                Shape::new(0.001, 0.01, 0.0, 0.05)
            );
            voice.add_processor(Box::new(High::new(200.0, rate)));
            voice.add_processor(Box::new(Crush::new(3, 4)));
            synth.add_voice(voice);
        }
        synth
    }

    pub fn melody_sequence() -> Sequence {
        let mut seq = Sequence::new(8.0, 140.0);
        seq.add(Pitch::from_midi(72, 0.8, 0.3, 0.0));
        seq.add(Pitch::from_midi(74, 0.8, 0.3, 0.5));
        seq.add(Pitch::from_midi(76, 0.8, 0.3, 1.0));
        seq.add(Pitch::from_midi(77, 0.8, 0.3, 1.5));
        seq.add(Pitch::from_midi(79, 0.9, 0.5, 2.0));
        seq.add(Pitch::from_midi(77, 0.7, 0.3, 2.75));
        seq.add(Pitch::from_midi(76, 0.8, 0.3, 3.25));
        seq.add(Pitch::from_midi(74, 0.8, 0.5, 3.75));
        seq.add(Pitch::from_midi(72, 0.8, 0.3, 4.5));
        seq.add(Pitch::from_midi(69, 0.8, 0.3, 5.0));
        seq.add(Pitch::from_midi(72, 0.8, 0.3, 5.5));
        seq.add(Pitch::from_midi(74, 0.8, 0.3, 6.0));
        seq.add(Pitch::from_midi(76, 0.9, 1.0, 6.5));
        seq
    }

    pub fn bass_sequence() -> Sequence {
        let mut seq = Sequence::new(8.0, 140.0);
        seq.add(Pitch::from_midi(36, 0.9, 0.4, 0.0));
        seq.add(Pitch::from_midi(36, 0.7, 0.2, 0.5));
        seq.add(Pitch::from_midi(43, 0.8, 0.4, 1.0));
        seq.add(Pitch::from_midi(41, 0.8, 0.4, 2.0));
        seq.add(Pitch::from_midi(38, 0.8, 0.4, 3.0));
        seq.add(Pitch::from_midi(36, 0.9, 0.4, 4.0));
        seq.add(Pitch::from_midi(36, 0.7, 0.2, 4.5));
        seq.add(Pitch::from_midi(33, 0.8, 0.4, 5.0));
        seq.add(Pitch::from_midi(36, 0.8, 0.4, 6.0));
        seq.add(Pitch::from_midi(38, 0.8, 0.4, 7.0));
        seq
    }

    pub fn arp_sequence() -> Sequence {
        let mut seq = Sequence::new(8.0, 140.0);
        let notes = [60, 64, 67, 72];
        for i in 0..32 {
            let note = notes[i % 4];
            let time = i as f32 * 0.25;
            seq.add(Pitch::from_midi(note, 0.6, 0.2, time));
        }
        seq
    }

    pub fn drum_sequence() -> Sequence {
        let mut seq = Sequence::new(8.0, 140.0);
        for i in 0..8 {
            let time = i as f32;
            if i % 2 == 0 {
                seq.add(Pitch::from_midi(60, 0.9, 0.1, time));
            } else {
                seq.add(Pitch::from_midi(80, 0.7, 0.05, time));
            }
        }
        seq
    }
}

pub fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let rate = 44100;
    let buffer_size = 512;

    let mut studio = Studio::new(rate, buffer_size);

    studio.add_device(Builder::melody_synth(rate));
    studio.add_device(Builder::bass_synth(rate));
    studio.add_device(Builder::arp_synth(rate));
    studio.add_device(Builder::drum_synth(rate));

    {
        let timeline = studio.get_timeline();
        if let Ok(mut timeline) = timeline.lock() {
            timeline.add_sequence("melody".to_string(), Builder::melody_sequence());
            timeline.add_sequence("bass".to_string(), Builder::bass_sequence());
            timeline.add_sequence("arp".to_string(), Builder::arp_sequence());
            timeline.add_sequence("drums".to_string(), Builder::drum_sequence());
            timeline.play("melody");
        }
    }

    studio.start()?;

    println!("Playing music... Press Enter to stop");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    studio.stop();
    Ok(())
}

fn main() {
    match demo() {
        Ok(_) => println!("Demo completed!"),
        Err(err) => println!("Error: {}", err),
    }
}