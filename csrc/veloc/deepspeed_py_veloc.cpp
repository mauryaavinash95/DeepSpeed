#include <deepspeed_py_veloc.h>

void veloc_ckpt_t::ckpt(ckpt_map_t &m, std::string path) {
    c10::impl::GenericDict generic_dict(c10::StringType::get(), m.begin()->second.type());
    for (const auto& entry : m) {
        generic_dict.insert(entry.first, entry.second);
    }
    torch::serialize::OutputArchive output_archive;
    output_archive.write("data", generic_dict);
    output_archive.save_to(path);
    std::cout << "Saving complete from CPP size for " << path << std::endl;
    return;
}

void veloc_ckpt_t::wait(size_t tensor_id) {
    std::cout << "Not implemented yet" << std::endl;
}
