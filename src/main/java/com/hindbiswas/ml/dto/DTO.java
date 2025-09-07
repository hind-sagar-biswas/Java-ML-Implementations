package com.hindbiswas.ml.dto;

/**
 * Base Data Transfer Object (DTO) for model serialization.
 * Contains versioning information for library and schema.
 */
public abstract class DTO {
    /** Library version for compatibility. */
    public String libraryVersion = "1.0.0";
    /** Schema version for DTO structure. */
    public String schemaVersion = "1";
}
