use std::collections::HashMap;
use std::fmt;

/// Represents the fundamental force that replaced conventional physics
#[derive(Debug, Clone, PartialEq)]
pub struct Resonance {
    pub frequency: f64,
    pub amplitude: f64,
    pub stability: f32,
}

/// Crystallized memories of lost realities
#[derive(Debug, Clone)]
pub struct EchoFragment {
    pub id: String,
    pub origin_reality: String,
    pub memory_content: String,
    pub power_level: u32,
    pub resonance: Resonance,
}

/// Ancient artifacts that stabilize regions
#[derive(Debug, Clone)]
pub struct RealityAnchor {
    pub name: String,
    pub location: Location,
    pub stability_radius: f64,
    pub is_active: bool,
    pub resonance_signature: Resonance,
}

/// Geographic locations in the shattered realms
#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    Luminaris,
    RustWastes,
    VerdantExpanse,
    ShiftingIsles,
    ObsidianDepths,
    EtherealSea,
    Unknown,
}

/// Character abilities and traits
#[derive(Debug, Clone)]
pub enum Ability {
    PatternWeaving,
    TechnologicalInterface,
    CouncilAuthority,
    AncientMemory,
    RealityManipulation,
}

/// Main character in the Nexus Arcana universe
#[derive(Debug, Clone)]
pub struct Character {
    pub name: String,
    pub background: String,
    pub current_location: Location,
    pub abilities: Vec<Ability>,
    pub resonance_attunement: Option<Resonance>,
    pub echo_fragments: Vec<EchoFragment>,
}

impl Character {
    /// Creates a new character with basic attributes
    pub fn new(name: String, background: String, location: Location) -> Self {
        Self {
            name,
            background,
            current_location: location,
            abilities: Vec::new(),
            resonance_attunement: None,
            echo_fragments: Vec::new(),
        }
    }

    /// Adds an ability to the character
    pub fn add_ability(&mut self, ability: Ability) {
        if !self.abilities.contains(&ability) {
            self.abilities.push(ability);
        }
    }

    /// Sets the character's resonance attunement
    pub fn attune_to_resonance(&mut self, resonance: Resonance) {
        self.resonance_attunement = Some(resonance);
    }

    /// Moves character to a new location
    pub fn travel_to(&mut self, destination: Location) {
        self.current_location = destination;
    }

    /// Adds an echo fragment to the character's collection
    pub fn collect_echo_fragment(&mut self, fragment: EchoFragment) {
        self.echo_fragments.push(fragment);
    }

    /// Calculates the character's total power level based on fragments
    pub fn total_power_level(&self) -> u32 {
        self.echo_fragments.iter().map(|f| f.power_level).sum()
    }

    /// Checks if character can manipulate a given resonance frequency
    pub fn can_manipulate_frequency(&self, target_frequency: f64) -> bool {
        if let Some(ref attunement) = self.resonance_attunement {
            let frequency_diff = (attunement.frequency - target_frequency).abs();
            frequency_diff <= attunement.amplitude * attunement.stability as f64
        } else {
            false
        }
    }

    /// Attempts to stabilize an unraveling event
    pub fn stabilize_unraveling(&self, unraveling_strength: f32) -> Result<f32, String> {
        if !self.abilities.contains(&Ability::PatternWeaving) {
            return Err("Character lacks pattern weaving ability".to_string());
        }

        if let Some(ref resonance) = self.resonance_attunement {
            let stabilization_power = resonance.amplitude as f32 * resonance.stability;
            if stabilization_power >= unraveling_strength {
                Ok(stabilization_power - unraveling_strength)
            } else {
                Err("Insufficient power to stabilize unraveling".to_string())
            }
        } else {
            Err("No resonance attunement found".to_string())
        }
    }

    /// Interfaces with ancient technology (for characters with TechnologicalInterface ability)
    pub fn interface_with_technology(&self, tech_complexity: u32) -> Result<String, String> {
        if !self.abilities.contains(&Ability::TechnologicalInterface) {
            return Err("Character cannot interface with technology".to_string());
        }

        let power = self.total_power_level();
        if power >= tech_complexity {
            Ok(format!("Successfully interfaced with technology (complexity: {})", tech_complexity))
        } else {
            Err(format!("Technology too complex (required: {}, available: {})", tech_complexity, power))
        }
    }

    /// Searches for reality anchors in the character's current location
    pub fn search_for_reality_anchors(&self, anchors: &[RealityAnchor]) -> Vec<&RealityAnchor> {
        anchors.iter()
            .filter(|anchor| anchor.location == self.current_location && anchor.is_active)
            .collect()
    }

    /// Attempts to communicate with the Verdant Expanse consciousness
    pub fn communicate_with_verdant_expanse(&self) -> Result<String, String> {
        if self.current_location != Location::VerdantExpanse {
            return Err("Must be in Verdant Expanse to communicate".to_string());
        }

        if self.abilities.contains(&Ability::PatternWeaving) {
            Ok("The forest consciousness responds through resonance patterns".to_string())
        } else {
            Ok("You sense a vast intelligence but cannot communicate directly".to_string())
        }
    }

    /// Calculates resonance compatibility with other characters
    pub fn resonance_compatibility(&self, other: &Character) -> f32 {
        match (&self.resonance_attunement, &other.resonance_attunement) {
            (Some(r1), Some(r2)) => {
                let freq_diff = (r1.frequency - r2.frequency).abs();
                let amp_ratio = (r1.amplitude / r2.amplitude).min(r2.amplitude / r1.amplitude);
                let stability_avg = (r1.stability + r2.stability) / 2.0;
                
                (amp_ratio as f32 * stability_avg) / (1.0 + freq_diff as f32)
            }
            _ => 0.0,
        }
    }

    /// Attempts to access fragmented memories (for Krell-type characters)
    pub fn access_fragmented_memory(&self, memory_key: &str) -> Option<String> {
        if !self.abilities.contains(&Ability::AncientMemory) {
            return None;
        }

        // Simulate fragmented memory access
        match memory_key {
            "convergence" => Some("Fragments of the Great Convergence event...".to_string()),
            "pre_convergence" => Some("Memories of the world before reality shattered...".to_string()),
            "anchors" => Some("Ancient knowledge of Reality Anchor locations...".to_string()),
            _ => None,
        }
    }

    /// Performs council authority actions (for Archon characters)
    pub fn exercise_council_authority(&self, action: &str) -> Result<String, String> {
        if !self.abilities.contains(&Ability::CouncilAuthority) {
            return Err("Character lacks council authority".to_string());
        }

        match action {
            "access_archives" => Ok("Accessing restricted Council archives...".to_string()),
            "mobilize_guards" => Ok("Archon Guards mobilized".to_string()),
            "seal_location" => Ok("Location sealed by Council decree".to_string()),
            _ => Err("Unknown authority action".to_string()),
        }
    }

    /// Manifests encounter with The Cipher entity
    pub fn encounter_cipher(&self) -> String {
        let manifestation = match self.current_location {
            Location::Luminaris => "A shimmering council member appears",
            Location::RustWastes => "An ancient machine consciousness emerges",
            Location::VerdantExpanse => "The forest itself seems to take shape",
            Location::ShiftingIsles => "A temporal echo phases into existence",
            Location::ObsidianDepths => "Crystalline formations arrange into a form",
            Location::EtherealSea => "The void itself gains awareness",
            Location::Unknown => "Reality bends around an impossible presence",
        };
        
        format!("The Cipher appears: {}", manifestation)
    }

    /// Handles dream sequences with The Cipher (for Lyra-type characters)
    pub fn cipher_dream_sequence(&mut self) -> Result<String, String> {
        if !self.abilities.contains(&Ability::PatternWeaving) {
            return Err("Only Pattern Weavers can experience Cipher dreams".to_string());
        }

        // Enhance abilities through dream teaching
        if let Some(ref mut resonance) = self.resonance_attunement {
            resonance.stability *= 1.1; // Improve stability
            resonance.amplitude *= 1.05; // Slight amplitude increase
        }

        Ok("The Cipher teaches advanced pattern weaving techniques in your dreams".to_string())
    }

    /// Investigates corruption within the Council (for exiled characters)
    pub fn investigate_corruption(&self, evidence_fragments: &[EchoFragment]) -> Vec<String> {
        let mut findings = Vec::new();
        
        for fragment in evidence_fragments {
            if fragment.origin_reality.contains("Council") || 
               fragment.memory_content.contains("harvest") {
                findings.push(format!("Suspicious activity detected: {}", fragment.memory_content));
            }
        }
        
        findings
    }

    /// Attempts to reverse engineer ancient technology
    pub fn reverse_engineer_tech(&self, tech_fragment: &EchoFragment) -> Result<String, String> {
        if !self.abilities.contains(&Ability::TechnologicalInterface) {
            return Err("Lacks technological interface capability".to_string());
        }

        if tech_fragment.power_level > self.total_power_level() {
            return Err("Technology too advanced to reverse engineer".to_string());
        }

        Ok(format!("Successfully reverse engineered: {}", tech_fragment.memory_content))
    }
}

impl fmt::Display for Character {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} from {} (currently in {:?})", 
               self.name, self.background, self.current_location)
    }
}

/// Creates the main characters from the Nexus Arcana universe
pub fn create_main_characters() -> HashMap<String, Character> {
    let mut characters = HashMap::new();

    // Lyra Vex - Pattern Weaver
    let mut lyra = Character::new(
        "Lyra Vex".to_string(),
        "Born in Luminaris, raised in Rust Wastes after parents disappeared".to_string(),
        Location::RustWastes,
    );
    lyra.add_ability(Ability::PatternWeaving);
    lyra.attune_to_resonance(Resonance {
        frequency: 440.0,
        amplitude: 2.5,
        stability: 0.8,
    });
    characters.insert("lyra".to_string(), lyra);

    // Thorn Blackwood - Exiled Archon Guard
    let mut thorn = Character::new(
        "Thorn Blackwood".to_string(),
        "Former Archon Guard exiled for discovering Council corruption".to_string(),
        Location::ObsidianDepths,
    );
    thorn.add_ability(Ability::TechnologicalInterface);
    thorn.attune_to_resonance(Resonance {
        frequency: 220.0,
        amplitude: 3.0,
        stability: 0.9,
    });
    characters.insert("thorn".to_string(), thorn);

    // Elysia Voss - Young Council Member
    let mut elysia = Character::new(
        "Elysia Voss".to_string(),
        "Youngest Council of Archons member, secretly researching Convergence reversal".to_string(),
        Location::Luminaris,
    );
    elysia.add_ability(Ability::CouncilAuthority);
    elysia.add_ability(Ability::RealityManipulation);
    characters.insert("elysia".to_string(), elysia);

    // Krell - Sentient Construct
    let mut krell = Character::new(
        "Krell".to_string(),
        "Sentient construct with fragmented pre-Convergence memories".to_string(),
        Location::ShiftingIsles,
    );
    krell.add_ability(Ability::AncientMemory);
    krell.add_ability(Ability::TechnologicalInterface);
    characters.insert("krell".to_string(), krell);

    characters
}

/// Creates sample reality anchors across the realms
pub fn create_reality_anchors() -> Vec<RealityAnchor> {
    vec![
        RealityAnchor {
            name: "Nexus Spire Core".to_string(),
            location: Location::Luminaris,
            stability_radius: 10000.0,
            is_active: true,
            resonance_signature: Resonance {
                frequency: 528.0,
                amplitude: 5.0,
                stability: 1.0,
            },
        },
        RealityAnchor {
            name: "Ancient Machine Heart".to_string(),
            location: Location::RustWastes,
            stability_radius: 5000.0,
            is_active: false,
            resonance_signature: Resonance {
                frequency: 110.0,
                amplitude: 4.0,
                stability: 0.6,
            },
        },
        RealityAnchor {
            name: "World Tree Root System".to_string(),
            location: Location::VerdantExpanse,
            stability_radius: 8000.0,
            is_active: true,
            resonance_signature: Resonance {
                frequency: 256.0,
                amplitude: 3.5,
                stability: 0.95,
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_creation() {
        let character = Character::new(
            "Test Character".to_string(),
            "Test background".to_string(),
            Location::Luminaris,
        );
        assert_eq!(character.name, "Test Character");
        assert_eq!(character.current_location, Location::Luminaris);
    }

    #[test]
    fn test_resonance_compatibility() {
        let mut char1 = Character::new("C1".to_string(), "B1".to_string(), Location::Luminaris);
        let mut char2 = Character::new("C2".to_string(), "B2".to_string(), Location::Luminaris);
        
        char1.attune_to_resonance(Resonance { frequency: 440.0, amplitude: 2.0, stability: 0.8 });
        char2.attune_to_resonance(Resonance { frequency: 440.0, amplitude: 2.0, stability: 0.8 });
        
        let compatibility = char1.resonance_compatibility(&char2);
        assert!(compatibility > 0.0);
    }
}