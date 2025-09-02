package com.hindbiswas.ml.util;

import java.lang.reflect.Constructor;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.hindbiswas.ml.dto.DTO;
import com.hindbiswas.ml.models.Model;

/**
 * Utility class for saving and loading {@link Model} instances to and from JSON
 * files.
 * 
 * <p>
 * This class handles:
 * <ul>
 * <li>Exporting models as JSON files.</li>
 * <li>Importing models back from JSON files using their corresponding
 * DTOs.</li>
 * </ul>
 */
public class ModelIO {

    /**
     * Exports the given {@link Model} to a specified file path as JSON.
     *
     * @param path  the path where the model should be saved
     * @param model the model instance to export
     * @return {@code true} if the export was successful, {@code false} otherwise
     */
    public static boolean export(Path path, Model model) {
        System.out.println("Exporting model to " + path);
        try {
            String json = model.toString();
            Files.write(path, json.getBytes(StandardCharsets.UTF_8));
            System.out.println("Model exported to " + path);
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed to export model to " + path + ": " + e.getMessage());
            System.err.println("Model: " + model);
            return false;
        }
    }

    /**
     * Imports a {@link Model} from a JSON file using its associated {@link DTO}.
     *
     * <p>
     * The method attempts to construct the model using a constructor that
     * accepts the provided {@code dtoClass} type.
     * </p>
     *
     * @param <D>        the type of the DTO used for deserialization
     * @param <M>        the type of the Model to be created
     * @param path       the path to the JSON file containing the serialized model
     * @param dtoClass   the class type of the DTO
     * @param modelClass the class type of the Model to instantiate
     * @return an instance of the deserialized model
     * @throws Exception if the file cannot be read, parsing fails, or no suitable
     *                   constructor is found
     */
    public static <D extends DTO, M extends Model> M importModel(Path path, Class<D> dtoClass, Class<M> modelClass)
            throws Exception {
        System.out.println("Importing model from " + path);
        String json = Files.readString(path, StandardCharsets.UTF_8);
        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        D dto = gson.fromJson(json, dtoClass);

        try {
            Constructor<M> ctor = modelClass.getConstructor(dtoClass);
            return ctor.newInstance(dto);
        } catch (NoSuchMethodException e) {
            for (Constructor<?> ctor : modelClass.getConstructors()) {
                Class<?>[] params = ctor.getParameterTypes();
                if (params.length == 1 && params[0].isAssignableFrom(dtoClass)) {
                    @SuppressWarnings("unchecked")
                    M instance = (M) ctor.newInstance(dto);
                    return instance;
                }
            }
            throw new NoSuchMethodException("No suitable constructor found in " + modelClass.getName()
                    + " that accepts " + dtoClass.getName());
        }
    }
}
