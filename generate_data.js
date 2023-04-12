const fs = require("fs");
const migrate = require("json-schema-migrate");
const path = require("path");
//const JSON5 = require("json5")
const jsf = require('@michaelmior/json-schema-faker');
jsf.extend('faker', () => require('@faker-js/faker'));
// Prevent the generation of additional properties
jsf.option({'noAdditional': true, 'requiredOnly': true});
// Prevent the creation of empty arrays or objects
//jsf.option({alwaysFakeOptionals: true, minItems: 1, fixedProbabilities: true});
let count = 0;
let fail = 0;


const Ajv2019 = require("ajv/dist/2019")
const ajv = new Ajv2019({strict:false, removeAdditional: true});
const addFormats = require('ajv-formats').default;
addFormats(ajv);
const draft7MetaSchema = require("ajv/dist/refs/json-schema-draft-07.json")
ajv.addMetaSchema(draft7MetaSchema)
const util = require("util")

/**
* Input: valid file, json document
* Output: None
* Purpose: Write json document to a file
*/
function write_data(file, json_data)
{
    // Stringify json object
    const json_content = JSON.stringify(json_data);
    // Write json object to file
    fs.appendFileSync(file, json_content + "\n", "utf-8", function (err){
        if (err){
            console.log("An error occured while writing json object to file.");
        }
    });//end file append
}


/**
* Input: curent folder 
* Output: new folder 
* Purpose: Create a new folder
*/
function create_folder(new_folder)
{
    try{
        // Delete folder if it exists
        if (fs.existsSync(new_folder)){
            fs.rmSync(new_folder, {recursive: true})
            console.log(`${new_folder} is deleted.`);
        }
        // Create a folder to store validated files
        fs.mkdirSync(new_folder);
    } 
    catch (err){
        console.error(`Error while deleting ${new_folder}.`);
    }
}


/**
* Input: schema
* Output: valid state (true or false)
* Purpose: Validate json schema
*/
function validate_schema(json_schema)
{
    try{
        // Validate json schema
        validate = ajv.validateSchema(json_schema);
        return validate;
    }
    catch(error){
        console.log(error);
        return false;
    }
}//end validate_schema   




/**
* Input: valid schema name, valid schema
* Output: a folder with valid json files
* Purpose: Validate json files
*/
function validate_file(schema_name, valid_schema)
{
    // Create a file to write the JSON documents generated from a schema
    file = valid_files_folder + schema_name
    
    let generated_documents = 0;
    let errors = 0;

    // Loop 1000 times, which represents the number of files generated
    while(generated_documents < 5)
    {
        let json_data;
        let valid;
        let schema = JSON.parse(JSON.stringify(valid_schema));
        
        try{
            // Generate json data from valid schema
            json_data = jsf.generate(schema);
            schema = valid_schema;
            // Check if json data is a non empty dictionary
            if (Object.prototype.toString.call(json_data) === "[object Object]" && Object.keys(json_data).length > 0){
                // Validate json document
                valid = ajv.validate(schema, json_data);
            }
        }
        catch(error){
            console.log(error)
            errors++;
        }
        
        // Check if the json data is invalid
        if (valid == true){
            write_data(file, json_data);
            console.log(schema_name, String(generated_documents), "written.");
            generated_documents++;
        }
    }//end while loop
    
    return generated_documents;

}//end validate_file


    
// Create a folder to store valid files
valid_files_folder = "files/";
create_folder(valid_files_folder);
// Get all schemas
const schemas = fs.readdirSync("schemas/");

// Loop through schemas
for (schema_name of schemas){
    // Get the paths of schema and associated files
    let schema = "schemas/" + schema_name;
    // Load json schema
    json_schema = JSON.parse(fs.readFileSync(schema));

    // Validate schema
    if(validate_schema(json_schema)){
        generated_documents = validate_file(schema_name, json_schema);
    }
    else{
        console.log(schema_name, "is invalid.", bad, '\n');
    }
}//end for loop

