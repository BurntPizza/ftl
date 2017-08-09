
extern crate ftl;

use std::fs;
use std::process;
use std::io::prelude::*;

#[test]
fn main() {
    for de in fs::read_dir("./tests")
        .unwrap()
        .map(Result::unwrap)
        .filter(|de| de.file_name().to_string_lossy().ends_with(".test"))
    {
        let mut file = fs::File::open(de.path()).unwrap();
        let mut input = String::new();
        file.read_to_string(&mut input).unwrap();

        let path = format!("{}", de.path().display());

        if input.lines().any(
            |l| l.to_lowercase().starts_with("// skip"),
        )
        {
            println!("Skipping test: {}", path);
            continue;
        }

        println!("Running test: {}", path);

        let expected_str = input
            .lines()
            .filter(|l| l.to_lowercase().starts_with("// out"))
            .map(ToOwned::to_owned)
            .next()
            .unwrap();


        let expected_str = expected_str[expected_str.to_lowercase().find("out").unwrap() + 3..]
            .trim();
        let input: String = input.lines().filter(|l| !l.starts_with("//")).collect();

        let prog = ftl::parser::parse_program(&*input).unwrap();
        let asm = ftl::compiler::codegen(ftl::compiler::compile(&prog));

        let asm_file_path = de.path().with_extension("s");
        let object_file_path = de.path().with_extension("o");
        let exec_file_path = de.path().with_extension("out");

        {
            let mut asm_file = fs::File::create(&asm_file_path).unwrap();
            asm_file.write_all(asm.as_bytes()).unwrap();
        }

        let status = process::Command::new("nasm")
            .arg("-felf64")
            .arg(&asm_file_path)
            .spawn()
            .expect("failed to create process nasm")
            .wait()
            .unwrap();

        assert!(status.success());

        let status = process::Command::new("ld")
            .arg(&object_file_path)
            .arg("-o")
            .arg(&exec_file_path)
            .spawn()
            .expect("failed to create process ld")
            .wait()
            .unwrap();

        assert!(status.success());

        let output = process::Command::new(&exec_file_path).output().expect(
            "failed to run test program",
        );

        assert!(output.status.success());

        let output = std::str::from_utf8(&*output.stdout)
            .unwrap()
            .trim()
            .replace("\n", " ");
        assert_eq!(output, expected_str);

        fs::remove_file(exec_file_path).unwrap();
        fs::remove_file(object_file_path).unwrap();
        fs::remove_file(asm_file_path).unwrap();
    }
}
